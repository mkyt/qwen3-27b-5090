// Tray app of the Windows package: starts the server (scripts/download_weights.py, then frontend/server.py, as
// start.cmd does) in a job object, so that it never outlives the app, shows its state from /health and its output in a
// log window. Decisions stay with the server: this app only starts it, stops it and reports. docs/configuration.md.
#ifndef UNICODE
#define UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <winhttp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

const wchar_t* const APP_TITLE = L"Qwen3.8-27B";
const wchar_t* const WINDOW_CLASS = L"qwen3-27b-5090-tray";
const wchar_t* const LOG_CLASS = L"qwen3-27b-5090-log";
const wchar_t* const MUTEX_NAME = L"Local\\qwen3-27b-5090-tray";
const wchar_t* const RUN_KEY = L"Software\\Microsoft\\Windows\\CurrentVersion\\Run";
const wchar_t* const RUN_VALUE = L"qwen3-27b-5090";

constexpr UINT WM_APP_TRAY = WM_APP + 1;     // from the notification icon
constexpr UINT WM_APP_CHANGED = WM_APP + 2;  // the shared state changed
constexpr UINT WM_APP_EXITED = WM_APP + 3;   // the thread that runs the server ended
constexpr UINT WM_APP_AGAIN = WM_APP + 4;    // another start of the app

enum MenuId { ID_STATUS = 1, ID_TOGGLE, ID_UI, ID_LOG, ID_LOGON, ID_QUIT, ID_LOG_FOLDER };

constexpr DWORD POLL_MS = 2000;
constexpr DWORD LOG_POLL_MS = 300;
constexpr LONGLONG LOG_ROTATE_BYTES = 16 << 20;  // the file, then one older file
constexpr LONGLONG LOG_TAIL_BYTES = 256 << 10;   // what the log window opens with
constexpr int LOG_WINDOW_CHARS = 1 << 20;        // the log window drops its older half beyond this

enum class Phase { stopped, preparing, downloading, running, stopping, failed };

struct Shared {  // written by the worker threads, read by the window's thread (mutex below)
    Phase phase = Phase::stopped;
    std::wstring failure;   // why it is Phase::failed
    std::wstring progress;  // the download's last progress line
    std::string health;     // "status" of /health ("": no answer yet)
    std::wstring refused;   // why the last load was refused (too little VRAM)
};

struct App {
    std::wstring root, python, server, download;     // the package's files
    std::wstring log_dir, log_path, old_log, ini;    // the package keeps what it writes in its own folder
    std::wstring host = L"127.0.0.1";
    int port = 8000;

    std::mutex mutex;
    Shared shared;
    HANDLE job = nullptr;       // of the running process (mutex)
    bool stop_wanted = false;   // (mutex)
    double idle_unload_s = 0;   // what the server last had (mutex)
    std::thread runner;         // while the server is started
    bool runner_active = false;
    HANDLE quit_event = nullptr;
    std::thread poller;

    HWND window = nullptr, log_window = nullptr, log_edit = nullptr;
    HFONT log_font = nullptr;
    LONGLONG log_offset = -1;   // -1: the log window opens at the tail
    NOTIFYICONDATAW icon{};
    UINT taskbar_created = 0;
    COLORREF icon_color = CLR_INVALID;
    HICON state_icon = nullptr;
    std::wstring shown_refused, shown_failure;
    bool download_shown = false;
    bool quitting = false;
};

App app;

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::string narrow(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

std::wstring upper(std::wstring s) {
    CharUpperBuffW(s.data(), static_cast<DWORD>(s.size()));
    return s;
}

bool exists(const std::wstring& path) { return GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES; }

void changed() { PostMessageW(app.window, WM_APP_CHANGED, 0, 0); }

// ------------------------------------------------------------ log file

class LogFile {
public:
    void open_new() {
        std::lock_guard lock(mutex_);
        reopen(true);
    }
    void line(const std::string& text) {
        std::lock_guard lock(mutex_);
        if (file_ == INVALID_HANDLE_VALUE) reopen(false);
        LARGE_INTEGER size;
        if (GetFileSizeEx(file_, &size) && size.QuadPart > LOG_ROTATE_BYTES) reopen(true);
        SYSTEMTIME t;
        GetLocalTime(&t);
        char stamp[16];
        std::snprintf(stamp, sizeof stamp, "%02d:%02d:%02d ", t.wHour, t.wMinute, t.wSecond);
        std::string out = stamp + text + "\r\n";
        DWORD written;
        WriteFile(file_, out.data(), static_cast<DWORD>(out.size()), &written, nullptr);
    }

private:
    void reopen(bool rotate) {
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        if (rotate) MoveFileExW(app.log_path.c_str(), app.old_log.c_str(), MOVEFILE_REPLACE_EXISTING);
        file_ = CreateFileW(app.log_path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    }
    std::mutex mutex_;
    HANDLE file_ = INVALID_HANDLE_VALUE;
};

LogFile logfile;

void tray_log(const std::string& text) { logfile.line("[tray] " + text); }

// ------------------------------------------------------------ settings

// A file's text as Notepad may save it: UTF-16, UTF-8 or the ANSI code page.
std::wstring read_text(const std::wstring& path) {
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return {};
    LARGE_INTEGER size{};
    GetFileSizeEx(file, &size);
    std::string raw(static_cast<size_t>(size.QuadPart), '\0');
    DWORD n = 0;
    ReadFile(file, raw.data(), static_cast<DWORD>(raw.size()), &n, nullptr);
    CloseHandle(file);
    raw.resize(n);
    if (raw.rfind("\xFF\xFE", 0) == 0) return std::wstring(reinterpret_cast<const wchar_t*>(raw.data() + 2), (raw.size() - 2) / 2);
    if (raw.rfind("\xEF\xBB\xBF", 0) == 0) raw.erase(0, 3);
    if (raw.empty()) return {};
    UINT page = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, raw.data(), static_cast<int>(raw.size()), nullptr, 0) ? CP_UTF8 : CP_ACP;
    int chars = MultiByteToWideChar(page, 0, raw.data(), static_cast<int>(raw.size()), nullptr, 0);
    std::wstring text(chars, L'\0');
    MultiByteToWideChar(page, 0, raw.data(), static_cast<int>(raw.size()), text.data(), chars);
    return text;
}

std::wstring trimmed(const std::wstring& s) {
    size_t first = s.find_first_not_of(L" \t\r\n"), last = s.find_last_not_of(L" \t\r\n");
    return first == std::wstring::npos ? std::wstring() : s.substr(first, last - first + 1);
}

std::vector<std::wstring> lines_of(const std::wstring& text) {
    std::vector<std::wstring> lines;
    size_t at = 0;
    while (at < text.size()) {
        size_t end = text.find(L'\n', at);
        if (end == std::wstring::npos) end = text.size();
        std::wstring line = text.substr(at, end - at);
        if (!line.empty() && line.back() == L'\r') line.pop_back();
        lines.push_back(line);
        at = end + 1;
    }
    return lines;
}

bool is_env_section(const std::wstring& line) { return _wcsicmp(trimmed(line).c_str(), L"[env]") == 0; }

// The name of a "name=value" line ("": a comment, a section or anything else).
std::wstring name_of(const std::wstring& raw) {
    std::wstring line = trimmed(raw);
    size_t eq = line.find(L'=');
    if (line.empty() || line[0] == L';' || line[0] == L'#' || line[0] == L'[' || eq == std::wstring::npos || eq == 0) return {};
    return upper(trimmed(line.substr(0, eq)));
}

// The [env] section of settings.ini by upper-case name.
std::map<std::wstring, std::wstring> read_settings() {
    std::map<std::wstring, std::wstring> settings;
    bool in_env = false;
    for (const auto& line : lines_of(read_text(app.ini))) {
        if (trimmed(line).rfind(L'[', 0) == 0) in_env = is_env_section(line);
        std::wstring name = name_of(line);
        if (!in_env || name.empty()) continue;
        std::wstring value = trimmed(line.substr(line.find(L'=') + 1));
        if (value.size() >= 2 && value.front() == L'"' && value.back() == L'"') value = value.substr(1, value.size() - 2);
        settings[name] = value;
    }
    return settings;
}

// Sets a name of the [env] section in settings.ini; the other lines stay as they are.
void write_setting(const std::wstring& name, const std::wstring& value) {
    std::vector<std::wstring> lines = lines_of(read_text(app.ini));
    size_t section = std::find_if(lines.begin(), lines.end(), is_env_section) - lines.begin();
    if (section == lines.size()) lines.push_back(L"[env]");
    size_t at = section + 1;
    while (at < lines.size() && trimmed(lines[at]).rfind(L'[', 0) != 0 && name_of(lines[at]) != upper(name)) at++;
    if (at < lines.size() && name_of(lines[at]) == upper(name))
        lines[at] = name + L"=" + value;
    else
        lines.insert(lines.begin() + section + 1, name + L"=" + value);
    std::wstring text;
    for (const auto& line : lines) text += line + L"\r\n";
    std::string utf8 = narrow(text);
    std::wstring temporary = app.ini + L".tmp";
    HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    DWORD written;
    WriteFile(file, utf8.data(), static_cast<DWORD>(utf8.size()), &written, nullptr);
    CloseHandle(file);
    MoveFileExW(temporary.c_str(), app.ini.c_str(), MOVEFILE_REPLACE_EXISTING);
}

// The server's environment: settings.ini, then this process's, then the defaults (and notes for the log). HF_HOME is
// the exception: only settings.ini moves the weights out of this folder, not a variable of the whole PC.
std::map<std::wstring, std::wstring> server_env(std::vector<std::string>* notes) {
    std::map<std::wstring, std::wstring> env;  // by upper-case name (names ignore case; Python's os.environ does the same)
    wchar_t* block = GetEnvironmentStringsW();
    for (const wchar_t* p = block; *p; p += wcslen(p) + 1) {
        std::wstring e = p;
        size_t eq = e.find(L'=', 1);
        if (eq != std::wstring::npos) env[upper(e.substr(0, eq))] = e.substr(eq + 1);
    }
    FreeEnvironmentStringsW(block);
    auto settings = read_settings();
    if (auto outside = env.find(L"HF_HOME"); outside != env.end()) {
        if (!settings.count(L"HF_HOME"))
            notes->push_back("HF_HOME=" + narrow(outside->second) + " in the environment is not used: set it in settings.ini to use that cache");
        env.erase(outside);
    }
    for (const auto& [k, v] : settings) {
        if (auto it = env.find(k); it != env.end() && it->second != v)
            notes->push_back(narrow(k + L"=" + v) + " from settings.ini (the environment has " + narrow(it->second) + ")");
        env[k] = v;
    }
    env.try_emplace(L"HOST", L"127.0.0.1");
    env.try_emplace(L"PORT", L"8000");
    env.try_emplace(L"ENGINE_AUTOSTART", L"0");         // no VRAM in use until the first request ...
    env.try_emplace(L"ENGINE_IDLE_UNLOAD_S", L"1800");  // ... and again after half an hour without one
    env.try_emplace(L"HF_HOME", L"huggingface");
    wchar_t full[MAX_PATH * 4];                         // (the working directory is the package's folder)
    if (GetFullPathNameW(env[L"HF_HOME"].c_str(), ARRAYSIZE(full), full, nullptr)) env[L"HF_HOME"] = full;
    env[L"PYTHONUNBUFFERED"] = L"1";  // the log follows the output
    env[L"PYTHONUTF8"] = L"1";        // the log is UTF-8 whatever the locale
    return env;
}

std::wstring env_block(const std::map<std::wstring, std::wstring>& env) {
    std::wstring block;
    for (const auto& [k, v] : env) block += k + L"=" + v + L'\0';  // (sorted, as CreateProcess documents)
    return block + L'\0';
}

// ------------------------------------------------------------ HTTP

// GET on the server's port; false when nothing answers.
bool http_get(HINTERNET session, const wchar_t* path, std::string* body) {
    bool ok = false;
    HINTERNET connection = WinHttpConnect(session, L"127.0.0.1", static_cast<INTERNET_PORT>(app.port), 0);
    HINTERNET request = connection ? WinHttpOpenRequest(connection, L"GET", path, nullptr, WINHTTP_NO_REFERER,
                                                        WINHTTP_DEFAULT_ACCEPT_TYPES, 0)
                                   : nullptr;
    if (request && WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
        WinHttpReceiveResponse(request, nullptr)) {
        ok = true;
        body->clear();
        DWORD n = 0;
        char buffer[4096];
        while (WinHttpReadData(request, buffer, sizeof buffer, &n) && n) body->append(buffer, n);
    }
    if (request) WinHttpCloseHandle(request);
    if (connection) WinHttpCloseHandle(connection);
    return ok;
}

HINTERNET http_session() {
    HINTERNET session = WinHttpOpen(L"qwen3-27b-5090-tray", WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME,
                                    WINHTTP_NO_PROXY_BYPASS, 0);
    if (session) WinHttpSetTimeouts(session, 1000, 1000, 1500, 1500);
    return session;
}

// The string value of `key` in a flat JSON object ("" when absent).
std::string json_string(const std::string& body, const std::string& key) {
    size_t at = body.find("\"" + key + "\"");
    if (at == std::string::npos) return {};
    at = body.find_first_not_of(" :", at + key.size() + 2);
    if (at == std::string::npos || body[at] != '"') return {};
    std::string out;
    for (size_t i = at + 1; i < body.size() && body[i] != '"'; i++) {
        if (body[i] != '\\' || i + 1 == body.size()) {
            out += body[i];
            continue;
        }
        char c = body[++i];
        out += c == 'n' ? '\n' : c == 't' ? '\t' : c;  // (\uXXXX: the server sends UTF-8, not escapes)
    }
    return out;
}

// The number value of `key` in a flat JSON object, as written ("" when absent).
std::string json_number(const std::string& body, const std::string& key) {
    size_t at = body.find("\"" + key + "\"");
    if (at == std::string::npos) return {};
    at = body.find_first_not_of(" :", at + key.size() + 2);
    if (at == std::string::npos) return {};
    return body.substr(at, body.find_first_of(",} ", at) - at);
}

// Keeps in settings.ini what the stats page changed (the server forgets it when it restarts): a value other than the
// one the server was started with or last reported.
void keep_idle_unload(const std::string& seconds) {
    double value = std::strtod(seconds.c_str(), nullptr);
    std::lock_guard lock(app.mutex);
    if (value == app.idle_unload_s) return;
    app.idle_unload_s = value;
    write_setting(L"ENGINE_IDLE_UNLOAD_S", widen(seconds));
}

void poll_loop() {
    HINTERNET session = http_session();
    while (WaitForSingleObject(app.quit_event, POLL_MS) == WAIT_TIMEOUT) {
        {
            std::lock_guard lock(app.mutex);
            if (app.shared.phase != Phase::running) continue;
        }
        std::string body, health;
        std::wstring refused;
        if (session && http_get(session, L"/health", &body)) {
            health = json_string(body, "status");
            refused = widen(json_string(body, "refused"));
            std::string idle = json_number(body, "idle_unload_s");
            if (!idle.empty()) keep_idle_unload(idle);
        }
        std::lock_guard lock(app.mutex);
        if (app.shared.phase == Phase::running && (app.shared.health != health || app.shared.refused != refused)) {
            app.shared.health = health;
            app.shared.refused = refused;
            changed();
        }
    }
    if (session) WinHttpCloseHandle(session);
}

// ------------------------------------------------------------ running the server

void set_phase(Phase phase, const std::wstring& failure = {}) {
    std::lock_guard lock(app.mutex);
    if (app.shared.phase == Phase::stopping && phase != Phase::stopped) return;
    app.shared.phase = phase;
    app.shared.failure = failure;
    if (phase != Phase::running) {
        app.shared.health.clear();
        app.shared.refused.clear();
    }
    changed();
}

void on_download_line(const std::string& text) {
    double done, total, rate;
    std::lock_guard lock(app.mutex);
    if (app.shared.phase == Phase::stopping) return;
    if (std::sscanf(text.c_str(), " %lf / %lf GB (%lf MB/s)", &done, &total, &rate) == 3) {
        wchar_t line[96];
        swprintf(line, 96, L"Downloading the weights: %.1f / %.1f GB (%.0f MB/s)", done, total, rate);
        app.shared.phase = Phase::downloading;
        app.shared.progress = line;
        changed();
    } else if (text.rfind("downloading ", 0) == 0) {
        app.shared.phase = Phase::downloading;
        app.shared.progress = L"Downloading the weights";
        changed();
    }
}

// Runs a script of the package to its end, in a job with all its descendants; the exit code (-1: it did not start).
template <class OnLine>
long run_python(const std::wstring& script, const std::wstring& args, const std::wstring& env, OnLine on_line) {
    SECURITY_ATTRIBUTES inherit{sizeof inherit, nullptr, TRUE};
    HANDLE read_end, write_end;
    if (!CreatePipe(&read_end, &write_end, &inherit, 0)) return -1;
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr);

    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof limits);

    STARTUPINFOW si{sizeof si};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul;
    si.hStdOutput = si.hStdError = write_end;
    PROCESS_INFORMATION pi{};
    std::wstring command = L"\"" + app.python + L"\" \"" + script + L"\"" + args;
    std::vector<wchar_t> line(command.begin(), command.end());
    line.push_back(L'\0');
    // Suspended until it is in the job: it starts the engine process, which has to be in the job too.
    BOOL started = CreateProcessW(app.python.c_str(), line.data(), nullptr, nullptr, TRUE,
                                  CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
                                  const_cast<wchar_t*>(env.c_str()), app.root.c_str(), &si, &pi);
    DWORD error = GetLastError();
    CloseHandle(write_end);
    CloseHandle(nul);
    if (!started) {
        tray_log("cannot start " + narrow(app.python) + ": error " + std::to_string(error));
        CloseHandle(read_end);
        CloseHandle(job);
        return -1;
    }
    AssignProcessToJobObject(job, pi.hProcess);
    bool stop;
    {
        std::lock_guard lock(app.mutex);
        app.job = job;
        stop = app.stop_wanted;
    }
    if (stop) TerminateJobObject(job, 1);
    ResumeThread(pi.hThread);

    std::thread reader([&] {
        std::string pending;
        char buffer[8192];
        DWORD n;
        auto emit = [&](std::string text) {
            if (!text.empty() && text.back() == '\r') text.pop_back();
            logfile.line(text);
            on_line(text);
        };
        while (ReadFile(read_end, buffer, sizeof buffer, &n, nullptr) && n) {
            pending.append(buffer, n);
            for (size_t at; (at = pending.find('\n')) != std::string::npos; pending.erase(0, at + 1))
                emit(pending.substr(0, at));
        }
        if (!pending.empty()) emit(pending);
    });
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    TerminateJobObject(job, 1);  // (what it left behind; the pipe then ends)
    reader.join();
    {
        std::lock_guard lock(app.mutex);
        app.job = nullptr;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(read_end);
    CloseHandle(job);
    return static_cast<long>(code);
}

bool stop_is_wanted() {
    std::lock_guard lock(app.mutex);
    return app.stop_wanted;
}

void run_server() {
    std::vector<std::string> notes;
    auto env = server_env(&notes);
    app.host = env[L"HOST"];
    app.port = _wtoi(env[L"PORT"].c_str());
    std::wstring block = env_block(env);
    {
        std::lock_guard lock(app.mutex);
        app.idle_unload_s = _wtof(env[L"ENGINE_IDLE_UNLOAD_S"].c_str());
    }
    logfile.open_new();
    tray_log("starting: HOST=" + narrow(app.host) + " PORT=" + std::to_string(app.port) + " HF_HOME=" + narrow(env[L"HF_HOME"]));
    for (const auto& note : notes) tray_log(note);

    // Windows lets two servers listen on one port (Python sets SO_REUSEADDR), so it has to be checked first.
    std::string body;
    HINTERNET session = http_session();
    bool taken = session && http_get(session, L"/health", &body);
    if (session) WinHttpCloseHandle(session);
    if (taken) {
        std::wstring why = L"Port " + std::to_wstring(app.port) + L" is in use by another server.";
        tray_log(narrow(why));
        set_phase(Phase::failed, why);
        PostMessageW(app.window, WM_APP_EXITED, 0, 0);
        return;
    }

    set_phase(Phase::preparing);
    long code = run_python(app.download, L"", block, on_download_line);
    if (!stop_is_wanted() && code != 0) {
        tray_log("the weights could not be downloaded (exit code " + std::to_string(code) + ")");
        set_phase(Phase::failed, L"The weights could not be downloaded. The log has the reason.");
    } else if (!stop_is_wanted()) {
        set_phase(Phase::running);
        code = run_python(app.server, L" --host " + app.host, block, [](const std::string&) {});
        if (!stop_is_wanted()) {
            tray_log("the server exited with code " + std::to_string(code));
            set_phase(Phase::failed, L"The server exited with code " + std::to_wstring(code) + L". The log has the reason.");
        }
    }
    if (stop_is_wanted()) {
        tray_log("stopped");
        set_phase(Phase::stopped);
    }
    PostMessageW(app.window, WM_APP_EXITED, 0, 0);
}

void start_server() {
    if (app.runner_active) return;
    {
        std::lock_guard lock(app.mutex);
        app.stop_wanted = false;
    }
    app.runner_active = true;
    app.runner = std::thread(run_server);
}

void stop_server() {
    std::lock_guard lock(app.mutex);
    if (!app.runner_active) return;
    app.stop_wanted = true;
    app.shared.phase = Phase::stopping;
    app.shared.health.clear();
    app.shared.refused.clear();
    if (app.job) TerminateJobObject(app.job, 1);
    changed();
}

// ------------------------------------------------------------ the notification icon

// The state as a ring with a tail (a Q) in that state's color, antialiased.
HICON make_icon(COLORREF color, int size) {
    BITMAPV5HEADER header{};
    header.bV5Size = sizeof header;
    header.bV5Width = size;
    header.bV5Height = -size;
    header.bV5Planes = 1;
    header.bV5BitCount = 32;
    header.bV5Compression = BI_BITFIELDS;
    header.bV5RedMask = 0x00FF0000;
    header.bV5GreenMask = 0x0000FF00;
    header.bV5BlueMask = 0x000000FF;
    header.bV5AlphaMask = 0xFF000000;
    void* bits = nullptr;
    HDC dc = GetDC(nullptr);
    HBITMAP bitmap = CreateDIBSection(dc, reinterpret_cast<BITMAPINFO*>(&header), DIB_RGB_COLORS, &bits, nullptr, 0);
    ReleaseDC(nullptr, dc);
    constexpr int SAMPLES = 4;
    auto inside = [](double x, double y) {  // (x, y in -1 .. 1)
        double r = std::sqrt(x * x + y * y);
        bool ring = r <= 0.94 && r >= 0.46;
        bool tail = x > 0.2 && y > 0.2 && std::abs(x - y) < 0.24 && x + y < 1.9;
        return ring || tail;
    };
    auto* out = static_cast<DWORD*>(bits);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int hits = 0;
            for (int j = 0; j < SAMPLES; j++)
                for (int i = 0; i < SAMPLES; i++)
                    hits += inside((x + (i + 0.5) / SAMPLES) * 2 / size - 1, (y + (j + 0.5) / SAMPLES) * 2 / size - 1);
            DWORD a = hits * 255 / (SAMPLES * SAMPLES);
            out[y * size + x] = (a << 24) | ((GetRValue(color) * a / 255) << 16) | ((GetGValue(color) * a / 255) << 8) |
                                (GetBValue(color) * a / 255);
        }
    }
    HBITMAP mask = CreateBitmap(size, size, 1, 1, nullptr);
    ICONINFO info{TRUE, 0, 0, mask, bitmap};
    HICON icon = CreateIconIndirect(&info);
    DeleteObject(mask);
    DeleteObject(bitmap);
    return icon;
}

struct View {
    COLORREF color;
    std::wstring text;
};

View view_of(const Shared& s) {
    const COLORREF gray = RGB(136, 136, 136), blue = RGB(64, 140, 230), amber = RGB(230, 165, 30), green = RGB(45, 175, 85),
                   orange = RGB(235, 110, 30), red = RGB(220, 50, 50);
    switch (s.phase) {
        case Phase::stopped: return {gray, L"Stopped"};
        case Phase::stopping: return {gray, L"Stopping"};
        case Phase::preparing: return {amber, L"Checking the weights"};
        case Phase::downloading: return {amber, s.progress};
        case Phase::failed: return {red, s.failure};
        case Phase::running: break;
    }
    if (!s.refused.empty()) return {orange, L"Not enough VRAM to load the model"};
    if (s.health == "ok") return {green, L"Ready"};
    if (s.health == "unloaded") return {blue, L"Waiting: the first request loads the model"};
    if (s.health == "loading") return {amber, L"Loading the model"};
    if (s.health == "stopped") return {gray, L"Engine stopped from the stats page"};
    if (s.health == "restarting") return {amber, L"Restarting the engine"};
    if (s.health == "stopping") return {amber, L"Stopping the engine"};
    return {amber, L"Starting the server"};
}

void copy_text(wchar_t* to, size_t size, const std::wstring& from) { wcsncpy_s(to, size, from.c_str(), _TRUNCATE); }

void add_icon() {
    Shell_NotifyIconW(NIM_ADD, &app.icon);
    app.icon.uVersion = NOTIFYICON_VERSION_4;
    Shell_NotifyIconW(NIM_SETVERSION, &app.icon);
}

void notify(const std::wstring& text, DWORD flags) {
    NOTIFYICONDATAW n = app.icon;
    n.uFlags = NIF_INFO;
    n.dwInfoFlags = flags;
    copy_text(n.szInfoTitle, ARRAYSIZE(n.szInfoTitle), APP_TITLE);
    copy_text(n.szInfo, ARRAYSIZE(n.szInfo), text);
    Shell_NotifyIconW(NIM_MODIFY, &n);
}

// Applies the shared state to the icon. Notifications only when the state changes (not at each refused request).
void refresh() {
    Shared s;
    {
        std::lock_guard lock(app.mutex);
        s = app.shared;
    }
    View view = view_of(s);
    if (view.color != app.icon_color) {
        HICON icon = make_icon(view.color, GetSystemMetrics(SM_CXSMICON));
        app.icon.hIcon = icon;
        app.icon.uFlags = NIF_ICON;
        Shell_NotifyIconW(NIM_MODIFY, &app.icon);
        if (app.state_icon) DestroyIcon(app.state_icon);
        app.state_icon = icon;
        app.icon_color = view.color;
    }
    app.icon.uFlags = NIF_TIP | NIF_SHOWTIP;
    copy_text(app.icon.szTip, ARRAYSIZE(app.icon.szTip), std::wstring(APP_TITLE) + L": " + view.text);
    Shell_NotifyIconW(NIM_MODIFY, &app.icon);

    if (s.phase == Phase::downloading && !app.download_shown) {
        app.download_shown = true;
        notify(L"Downloading the weights. The tooltip of this icon shows the progress; an interrupted download resumes.",
               NIIF_INFO);
    } else if (s.phase == Phase::running && app.download_shown) {
        app.download_shown = false;
        notify(L"The weights are downloaded.", NIIF_INFO);
    }
    if (s.failure != app.shown_failure) {
        app.shown_failure = s.failure;
        if (!s.failure.empty()) notify(s.failure, NIIF_ERROR);
    }
    if (s.refused != app.shown_refused) {
        app.shown_refused = s.refused;
        if (!s.refused.empty()) notify(s.refused, NIIF_WARNING);
    }
}

// ------------------------------------------------------------ the log window

void log_append(const std::string& utf8) {
    std::wstring text;
    for (wchar_t c : widen(utf8)) {  // (an edit control breaks lines at \r\n only)
        if (c == L'\r') continue;
        if (c == L'\n') text += L'\r';
        text += c;
    }
    HWND edit = app.log_edit;
    SCROLLINFO scroll{sizeof scroll, SIF_ALL};
    GetScrollInfo(edit, SB_VERT, &scroll);
    bool follow = !(GetWindowLongW(edit, GWL_STYLE) & WS_VSCROLL) || scroll.nMax == 0 ||
                  scroll.nPos + static_cast<int>(scroll.nPage) > scroll.nMax - 1;
    DWORD sel_start, sel_end;
    SendMessageW(edit, EM_GETSEL, reinterpret_cast<WPARAM>(&sel_start), reinterpret_cast<LPARAM>(&sel_end));
    int first_line = static_cast<int>(SendMessageW(edit, EM_GETFIRSTVISIBLELINE, 0, 0));
    SendMessageW(edit, WM_SETREDRAW, FALSE, 0);
    int length = GetWindowTextLengthW(edit);
    if (length + static_cast<int>(text.size()) > LOG_WINDOW_CHARS) {
        int line = static_cast<int>(SendMessageW(edit, EM_LINEFROMCHAR, length / 2, 0));
        int cut = static_cast<int>(SendMessageW(edit, EM_LINEINDEX, line, 0));
        SendMessageW(edit, EM_SETSEL, 0, cut);
        SendMessageW(edit, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(L""));
        follow = true;
        length = GetWindowTextLengthW(edit);
    }
    SendMessageW(edit, EM_SETSEL, length, length);
    SendMessageW(edit, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(text.c_str()));
    if (!follow) {
        SendMessageW(edit, EM_SETSEL, sel_start, sel_end);
        int now = static_cast<int>(SendMessageW(edit, EM_GETFIRSTVISIBLELINE, 0, 0));
        SendMessageW(edit, EM_LINESCROLL, 0, first_line - now);
    }
    SendMessageW(edit, WM_SETREDRAW, TRUE, 0);
    InvalidateRect(edit, nullptr, TRUE);
    if (follow) {
        int end = GetWindowTextLengthW(edit);
        SendMessageW(edit, EM_SETSEL, end, end);
        SendMessageW(edit, EM_SCROLLCARET, 0, 0);
    }
}

// Follows the file, not the running process: the log is what is left to read when the server did not start.
void log_poll() {
    HANDLE file = CreateFileW(app.log_path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    LARGE_INTEGER size{};
    GetFileSizeEx(file, &size);
    bool tail = app.log_offset < 0;
    if (tail) app.log_offset = std::max<LONGLONG>(0, size.QuadPart - LOG_TAIL_BYTES);
    if (size.QuadPart < app.log_offset) app.log_offset = 0;  // (rotated)
    tail = tail && app.log_offset > 0;
    if (size.QuadPart > app.log_offset) {
        std::string data(static_cast<size_t>(std::min<LONGLONG>(size.QuadPart - app.log_offset, 1 << 20)), '\0');
        LARGE_INTEGER at;
        at.QuadPart = app.log_offset;
        DWORD n = 0;
        SetFilePointerEx(file, at, nullptr, FILE_BEGIN);
        if (ReadFile(file, data.data(), static_cast<DWORD>(data.size()), &n, nullptr)) {
            data.resize(n);
            size_t end = data.rfind('\n');  // (whole lines: a character may be cut otherwise)
            if (end != std::string::npos) {
                data.resize(end + 1);
                app.log_offset += data.size();
                size_t start = tail ? data.find('\n') + 1 : 0;
                log_append(data.substr(start));
            }
        }
    }
    CloseHandle(file);
}

void set_log_font(HWND window) {
    HFONT font = CreateFontW(-MulDiv(10, GetDpiForWindow(window), 72), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                             OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FIXED_PITCH | FF_MODERN, L"Consolas");
    SendMessageW(app.log_edit, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    if (app.log_font) DeleteObject(app.log_font);
    app.log_font = font;
}

LRESULT CALLBACK log_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    switch (message) {
        case WM_CREATE:
            app.log_edit = CreateWindowExW(0, L"EDIT", nullptr,
                                           WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_HSCROLL | ES_MULTILINE | ES_READONLY |
                                               ES_AUTOVSCROLL | ES_AUTOHSCROLL | ES_NOHIDESEL,
                                           0, 0, 0, 0, window, nullptr, GetModuleHandleW(nullptr), nullptr);
            SendMessageW(app.log_edit, EM_SETLIMITTEXT, 0, 0);
            set_log_font(window);
            app.log_offset = -1;
            log_poll();
            SetTimer(window, 1, LOG_POLL_MS, nullptr);
            return 0;
        case WM_SIZE:
            MoveWindow(app.log_edit, 0, 0, LOWORD(lparam), HIWORD(lparam), TRUE);
            return 0;
        case WM_SETFOCUS:
            SetFocus(app.log_edit);
            return 0;
        case WM_TIMER:
            log_poll();
            return 0;
        case WM_DPICHANGED: {
            set_log_font(window);
            auto* r = reinterpret_cast<RECT*>(lparam);
            SetWindowPos(window, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
            return 0;
        }
        case WM_COMMAND:
            if (LOWORD(wparam) == ID_LOG_FOLDER)
                ShellExecuteW(window, L"open", app.log_dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            return 0;
        case WM_DESTROY:
            KillTimer(window, 1);
            app.log_window = app.log_edit = nullptr;
            return 0;
    }
    return DefWindowProcW(window, message, wparam, lparam);
}

void show_log() {
    if (!app.log_window) {
        HMENU menu = CreateMenu();
        AppendMenuW(menu, MF_STRING, ID_LOG_FOLDER, L"Open log &folder");
        std::wstring title = std::wstring(APP_TITLE) + L" log";
        UINT dpi = GetDpiForSystem();
        app.log_window = CreateWindowExW(0, LOG_CLASS, title.c_str(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                                         MulDiv(1000, dpi, 96), MulDiv(620, dpi, 96), nullptr, menu,
                                         GetModuleHandleW(nullptr), nullptr);
    }
    ShowWindow(app.log_window, IsIconic(app.log_window) ? SW_RESTORE : SW_SHOW);
    SetForegroundWindow(app.log_window);
}

// ------------------------------------------------------------ start at logon

std::wstring run_command() {
    wchar_t path[MAX_PATH * 4];
    GetModuleFileNameW(nullptr, path, ARRAYSIZE(path));
    return std::wstring(L"\"") + path + L"\"";
}

bool logon_value(std::wstring* value) {
    wchar_t buffer[MAX_PATH * 4 + 8];
    DWORD size = sizeof buffer;
    if (RegGetValueW(HKEY_CURRENT_USER, RUN_KEY, RUN_VALUE, RRF_RT_REG_SZ, nullptr, buffer, &size) != ERROR_SUCCESS) return false;
    *value = buffer;
    return true;
}

void set_logon(bool on) {
    HKEY key;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, RUN_KEY, 0, nullptr, 0, KEY_SET_VALUE, nullptr, &key, nullptr) != ERROR_SUCCESS) return;
    if (on) {
        std::wstring command = run_command();
        RegSetValueExW(key, RUN_VALUE, 0, REG_SZ, reinterpret_cast<const BYTE*>(command.c_str()),
                       static_cast<DWORD>((command.size() + 1) * sizeof(wchar_t)));
    } else {
        RegDeleteValueW(key, RUN_VALUE);
    }
    RegCloseKey(key);
}

// ------------------------------------------------------------ the menu and the main window

void show_menu() {
    Shared s;
    bool active;
    {
        std::lock_guard lock(app.mutex);
        s = app.shared;
        active = app.runner_active;
    }
    std::wstring ignored;
    HMENU menu = CreatePopupMenu();
    AppendMenuW(menu, MF_STRING | MF_GRAYED, ID_STATUS, view_of(s).text.c_str());
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_STRING | (s.phase == Phase::stopping ? MF_GRAYED : 0), ID_TOGGLE, active ? L"&Stop server" : L"&Start server");
    AppendMenuW(menu, MF_STRING | (s.phase == Phase::running ? 0 : MF_GRAYED), ID_UI, L"Open stats &page");
    AppendMenuW(menu, MF_STRING, ID_LOG, L"Show &log");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_STRING | (logon_value(&ignored) ? MF_CHECKED : 0), ID_LOGON, L"Start at log&on");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_STRING, ID_QUIT, L"&Quit");
    POINT at;
    GetCursorPos(&at);
    SetForegroundWindow(app.window);  // (or the menu stays open after a click elsewhere)
    TrackPopupMenuEx(menu, TPM_RIGHTBUTTON | TPM_BOTTOMALIGN, at.x, at.y, app.window, nullptr);
    PostMessageW(app.window, WM_NULL, 0, 0);
    DestroyMenu(menu);
}

void quit() {
    app.quitting = true;
    if (app.runner_active)
        stop_server();  // (WM_APP_EXITED goes on)
    else
        DestroyWindow(app.window);
}

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == app.taskbar_created && message) {  // (Explorer restarted)
        add_icon();
        return 0;
    }
    switch (message) {
        case WM_APP_TRAY:
            switch (LOWORD(lparam)) {
                case WM_CONTEXTMENU:
                case NIN_SELECT:
                case NIN_KEYSELECT: show_menu(); break;
                case NIN_BALLOONUSERCLICK: show_log(); break;
            }
            return 0;
        case WM_APP_CHANGED:
            refresh();
            return 0;
        case WM_APP_EXITED: {
            app.runner.join();
            {
                std::lock_guard lock(app.mutex);
                app.runner_active = false;
                if (app.shared.phase == Phase::stopping) app.shared.phase = Phase::stopped;
            }
            if (app.quitting)
                DestroyWindow(window);
            else
                refresh();
            return 0;
        }
        case WM_APP_AGAIN:
            notify(L"Already running: its menu is on this icon.", NIIF_INFO);
            return 0;
        case WM_COMMAND:
            switch (LOWORD(wparam)) {
                case ID_TOGGLE:
                    if (app.runner_active)
                        stop_server();
                    else
                        start_server();
                    break;
                case ID_UI: {
                    std::wstring url = L"http://127.0.0.1:" + std::to_wstring(app.port) + L"/ui";
                    ShellExecuteW(window, L"open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
                    break;
                }
                case ID_LOG: show_log(); break;
                case ID_LOGON: {
                    std::wstring ignored;
                    set_logon(!logon_value(&ignored));
                    break;
                }
                case ID_QUIT: quit(); break;
            }
            return 0;
        case WM_DESTROY:
            Shell_NotifyIconW(NIM_DELETE, &app.icon);
            if (app.log_window) DestroyWindow(app.log_window);
            SetEvent(app.quit_event);
            if (app.poller.joinable()) app.poller.join();
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(window, message, wparam, lparam);
}

bool find_package() {
    wchar_t path[MAX_PATH * 4];
    GetModuleFileNameW(nullptr, path, ARRAYSIZE(path));
    app.root = path;
    app.root.erase(app.root.find_last_of(L'\\'));
    app.python = app.root + L"\\python\\python.exe";
    app.server = app.root + L"\\app\\frontend\\server.py";
    app.download = app.root + L"\\app\\scripts\\download_weights.py";
    app.log_dir = app.root + L"\\logs";
    app.log_path = app.log_dir + L"\\server.log";
    app.old_log = app.log_dir + L"\\server.1.log";
    app.ini = app.root + L"\\settings.ini";
    return exists(app.python) && exists(app.server) && exists(app.download);
}

// What the package writes stays in its folder, so the folder has to take it.
bool can_write() {
    CreateDirectoryW(app.log_dir.c_str(), nullptr);
    HANDLE file = CreateFileW((app.log_dir + L"\\.write-test").c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    CloseHandle(file);
    return true;
}

}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR command_line, int) {
    bool found = find_package();
    bool writable = found && can_write();
    if (wcscmp(command_line, L"--check") == 0) return !found ? 2 : writable ? 0 : 3;  // (the package's smoke test)
    if (!found) {
        MessageBoxW(nullptr, L"python\\python.exe and app\\ were not found next to this program. Keep it in the folder of the unzipped package.",
                    APP_TITLE, MB_ICONERROR);
        return 2;
    }
    if (!writable) {
        std::wstring text = L"Cannot write in " + app.root + L".\n\nThis program keeps its settings, log and weights in its own folder. Unzip the package in a folder you can write to (not Program Files).";
        MessageBoxW(nullptr, text.c_str(), APP_TITLE, MB_ICONERROR);
        return 3;
    }
    SetCurrentDirectoryW(app.root.c_str());  // (a start at logon has another one; relative paths in settings.ini are from here)
    HANDLE mutex = CreateMutexW(nullptr, TRUE, MUTEX_NAME);
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        if (HWND other = FindWindowW(WINDOW_CLASS, nullptr)) PostMessageW(other, WM_APP_AGAIN, 0, 0);
        return 0;
    }

    std::wstring registered;  // (the package may have moved since the box was ticked)
    if (logon_value(&registered) && registered != run_command()) set_logon(true);

    WNDCLASSEXW window_class{sizeof window_class};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = instance;
    window_class.lpszClassName = WINDOW_CLASS;
    RegisterClassExW(&window_class);
    WNDCLASSEXW log_class{sizeof log_class};
    log_class.lpfnWndProc = log_proc;
    log_class.hInstance = instance;
    log_class.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(1));
    log_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    log_class.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    log_class.lpszClassName = LOG_CLASS;
    RegisterClassExW(&log_class);

    app.taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");
    app.quit_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    app.window = CreateWindowExW(0, WINDOW_CLASS, APP_TITLE, WS_OVERLAPPED, 0, 0, 0, 0, nullptr, nullptr, instance, nullptr);

    app.icon.cbSize = sizeof app.icon;
    app.icon.hWnd = app.window;
    app.icon.uID = 1;
    app.icon.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP | NIF_SHOWTIP;
    app.icon.uCallbackMessage = WM_APP_TRAY;
    app.icon_color = RGB(136, 136, 136);
    app.state_icon = app.icon.hIcon = make_icon(app.icon_color, GetSystemMetrics(SM_CXSMICON));
    copy_text(app.icon.szTip, ARRAYSIZE(app.icon.szTip), APP_TITLE);
    add_icon();

    app.poller = std::thread(poll_loop);
    start_server();

    MSG message;
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    CloseHandle(mutex);
    return 0;
}
