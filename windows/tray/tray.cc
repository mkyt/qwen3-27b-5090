// Tray app of the Windows package: starts the server (scripts/download_weights.py, then frontend/server.py, as
// start.cmd does) in a job object, so that it never outlives the app, shows its state from /health and its output in a
// log window, and edits settings.ini in a settings window. Decisions stay with the server: this app starts it, stops it,
// reports and keeps the file. docs/configuration.md.
#ifndef UNICODE
#define UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <objbase.h>
#include <shellapi.h>
#include <shobjidl.h>
#include <winhttp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cwchar>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

const wchar_t* const APP_TITLE = L"Qwen3.8-27B";
const wchar_t* const WINDOW_CLASS = L"qwen3-27b-5090-tray";
const wchar_t* const LOG_CLASS = L"qwen3-27b-5090-log";
const wchar_t* const SETTINGS_CLASS = L"qwen3-27b-5090-settings";
const wchar_t* const MUTEX_NAME = L"Local\\qwen3-27b-5090-tray";
const wchar_t* const RUN_KEY = L"Software\\Microsoft\\Windows\\CurrentVersion\\Run";
const wchar_t* const RUN_VALUE = L"qwen3-27b-5090";

constexpr UINT WM_APP_TRAY = WM_APP + 1;     // from the notification icon
constexpr UINT WM_APP_CHANGED = WM_APP + 2;  // the shared state changed
constexpr UINT WM_APP_EXITED = WM_APP + 3;   // the thread that runs the server ended
constexpr UINT WM_APP_AGAIN = WM_APP + 4;    // another start of the app
constexpr UINT WM_APP_SETTINGS = WM_APP + 5; // an answer for the settings window

// (update.ps1 quits the tray of its package with WM_COMMAND ID_QUIT to WINDOW_CLASS: change both together)
enum MenuId { ID_STATUS = 1, ID_TOGGLE, ID_UI, ID_LOG, ID_LOGON, ID_QUIT, ID_LOG_FOLDER, ID_SETTINGS };

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
    std::wstring root, python, server, download, plan;  // the package's files
    std::wstring log_dir, log_path, old_log, ini;    // the package keeps what it writes in its own folder
    std::wstring host = L"127.0.0.1";
    int port = 8000;

    std::mutex mutex;
    Shared shared;
    HANDLE job = nullptr;       // of the running process (mutex)
    bool stop_wanted = false;   // (mutex)
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
    bool restart_pending = false;  // the server starts again when it has stopped (the settings window)
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

// ------------------------------------------------------------ JSON (the admin API's answers)

struct Json {
    enum Type { null, boolean, number, string, array, object } type = null;
    bool flag = false;
    double value = 0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> fields;
    const Json& operator[](const std::string& key) const {
        static const Json none;
        for (const auto& field : fields)
            if (field.first == key) return field.second;
        return none;
    }
};

class JsonReader {
public:
    explicit JsonReader(const std::string& source) : s_(source) {}
    bool read(Json* out) { return value(out, 0); }

private:
    void space() {
        while (at_ < s_.size() && strchr(" \t\r\n", s_[at_])) at_++;
    }
    bool literal(const char* word) {
        size_t n = strlen(word);
        if (s_.compare(at_, n, word) != 0) return false;
        at_ += n;
        return true;
    }
    unsigned hex4() {
        unsigned code = 0;
        for (int i = 0; i < 4 && at_ < s_.size(); i++, at_++) {
            char c = s_[at_];
            code = code * 16 + (c <= '9' ? c - '0' : (c | 32) - 'a' + 10);
        }
        return code;
    }
    bool string(std::string* out) {
        if (at_ >= s_.size() || s_[at_] != '"') return false;
        for (at_++; at_ < s_.size(); at_++) {
            char c = s_[at_];
            if (c == '"') {
                at_++;
                return true;
            }
            if (c != '\\') {
                *out += c;
                continue;
            }
            if (++at_ == s_.size()) return false;
            c = s_[at_];
            if (c == 'u') {
                at_++;
                wchar_t units[2] = {static_cast<wchar_t>(hex4()), 0};
                if (units[0] >= 0xD800 && units[0] < 0xDC00 && s_.compare(at_, 2, "\\u") == 0) {
                    at_ += 2;
                    units[1] = static_cast<wchar_t>(hex4());
                }
                at_--;
                *out += narrow(units);
            } else {
                *out += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c == 'b' ? '\b' : c == 'f' ? '\f' : c;
            }
        }
        return false;
    }
    bool value(Json* out, int depth) {
        space();
        if (at_ >= s_.size() || depth > 32) return false;
        char c = s_[at_];
        if (c == '{' || c == '[') {
            out->type = c == '{' ? Json::object : Json::array;
            at_++;
            for (space(); at_ < s_.size() && s_[at_] != (c == '{' ? '}' : ']'); space()) {
                std::string key;
                if (c == '{') {
                    if (!string(&key)) return false;
                    space();
                    if (at_ >= s_.size() || s_[at_++] != ':') return false;
                }
                Json item;
                if (!value(&item, depth + 1)) return false;
                if (c == '{')
                    out->fields.emplace_back(std::move(key), std::move(item));
                else
                    out->items.push_back(std::move(item));
                space();
                if (at_ < s_.size() && s_[at_] == ',') at_++;
            }
            return at_++ < s_.size();
        }
        if (c == '"') {
            out->type = Json::string;
            return string(&out->text);
        }
        if (literal("null")) return true;
        if (literal("true") || literal("false")) {
            out->type = Json::boolean;
            out->flag = s_[at_ - 1] == 'e' && s_[at_ - 2] == 'u';
            return true;
        }
        char* end = nullptr;
        out->type = Json::number;
        out->value = strtod(s_.c_str() + at_, &end);
        if (end == s_.c_str() + at_) return false;
        at_ = end - s_.c_str();
        return true;
    }
    const std::string& s_;
    size_t at_ = 0;
};

std::string json_quote(const std::wstring& w) {
    std::string out = "\"";
    for (char c : narrow(w)) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

// A setting's value as its variable has it ("": null, the automatic budget).
std::wstring env_text(const Json& v) {
    wchar_t number[32];
    switch (v.type) {
        case Json::string: return widen(v.text);
        case Json::number: swprintf(number, 32, L"%g", v.value); return number;
        default: return {};
    }
}

// ------------------------------------------------------------ HTTP with a body

struct HttpAnswer {
    bool answered = false;
    DWORD status = 0;
    std::string body;
};

// A request to the server's port that may take a while (a plan: up to about a second; a restart's checks).
HttpAnswer http_call(int port, const wchar_t* method, const wchar_t* path, const std::string& body) {
    HttpAnswer answer;
    HINTERNET session = http_session();
    if (session) WinHttpSetTimeouts(session, 2000, 2000, 5000, 30000);
    HINTERNET connection = session ? WinHttpConnect(session, L"127.0.0.1", static_cast<INTERNET_PORT>(port), 0) : nullptr;
    HINTERNET request = connection ? WinHttpOpenRequest(connection, method, path, nullptr, WINHTTP_NO_REFERER,
                                                        WINHTTP_DEFAULT_ACCEPT_TYPES, 0)
                                   : nullptr;
    const wchar_t* header = L"Content-Type: application/json\r\n";
    if (request && WinHttpSendRequest(request, header, static_cast<DWORD>(-1), const_cast<char*>(body.data()),
                                      static_cast<DWORD>(body.size()), static_cast<DWORD>(body.size()), 0) &&
        WinHttpReceiveResponse(request, nullptr)) {
        answer.answered = true;
        DWORD size = sizeof answer.status;
        WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                            &answer.status, &size, WINHTTP_NO_HEADER_INDEX);
        DWORD n = 0;
        char buffer[8192];
        while (WinHttpReadData(request, buffer, sizeof buffer, &n) && n) answer.body.append(buffer, n);
    }
    if (request) WinHttpCloseHandle(request);
    if (connection) WinHttpCloseHandle(connection);
    if (session) WinHttpCloseHandle(session);
    return answer;
}

std::wstring api_error(const HttpAnswer& answer) {
    Json parsed;
    if (JsonReader(answer.body).read(&parsed)) {
        const Json& error = parsed["error"];
        if (error["message"].type == Json::string) return widen(error["message"].text);
        if (error.type == Json::string) return widen(error.text);
    }
    return answer.answered ? L"HTTP " + std::to_wstring(answer.status) : L"the server did not answer";
}

// ------------------------------------------------------------ the settings window

// An editor of settings.ini: the file is what stays, so Apply writes it and then does what the changed settings need
// (docs/configuration.md, "How settings work"). The choices and what each leaves for the context come from the server.

enum class Tier { engine, live, server };
enum class Kind { choice, edit, combo_edit };

struct Row {
    const wchar_t* label;
    const wchar_t* var;
    const char* key;    // of the start settings in the admin API (nullptr: not one)
    Kind kind;
    Tier tier;
    const wchar_t* hint;
    HWND label_window = nullptr, control = nullptr, hint_window = nullptr;
    std::vector<std::wstring> values;  // of a choice
    std::wstring initial;              // what settings.ini gives
    bool known = true;                 // false: neither the file nor the environment has it; the server's default, once it answers
    bool in_file = false;              // settings.ini has it
    bool chosen = false;               // the user chose or typed a value: it goes in the file, even if it equals the default
};

enum SettingControl { IDC_ROW = 100, IDC_APPLY = 200, IDC_CLOSE, IDC_OPEN_INI, IDC_USE_KEEP, IDC_FIX_LINKS, IDC_BROWSE };
enum { TIMER_TICK = 1, TIMER_PLAN = 2, TIMER_SCAN = 3 };

constexpr int ROW_SERVER_LAST = 2, ROW_ENGINE_LAST = 9, ROW_LOAD_LAST = 12;

struct SettingsWindow {
    HWND window = nullptr, state = nullptr, summary = nullptr, load_warning = nullptr, host_warning = nullptr, folder_note = nullptr;
    HWND browse = nullptr, fix_links = nullptr, keep_text = nullptr, keep_button = nullptr, message = nullptr, apply = nullptr, close = nullptr, open_ini = nullptr;
    HWND groups[3] = {};
    HFONT font = nullptr;
    std::vector<Row> rows;
    Json engine;                // the last GET /admin/engine
    bool has_engine = false;
    bool busy = false;          // a request is out
    bool plan_wanted = false;
    bool message_is_error = false, message_is_result = false, filling = false;
    bool restarting = false;    // Apply restarted the server: say so when it answers
    // What the weights folder in the window has (frontend/plan.py).
    bool scanning = false, scanned = false, scan_draft = true, download_needed = false, folder_warning = false;
    std::wstring scan_folder;
    std::vector<std::wstring> scan_weights, scan_links;
    std::map<std::wstring, std::wstring> keep;  // the stats page's changes that the controls do not have
} settings;

struct SettingsResult {
    enum What { engine, plan, applied, scan, fixed } what;
    HttpAnswer answer;
    std::wstring text;  // applied: what to tell; scan: the folder
    bool failed = false;
};

std::vector<Row> make_rows() {
    using K = Kind;
    using T = Tier;
    return {
        {L"Address", L"HOST", nullptr, K::combo_edit, T::server, nullptr},
        {L"Port", L"PORT", nullptr, K::edit, T::server, nullptr},
        {L"Weights folder (HF_HOME)", L"HF_HOME", nullptr, K::edit, T::server, nullptr},
        {L"Weights", L"WEIGHTS", "weights", K::choice, T::engine, nullptr},
        {L"KV cache type", L"KV_DTYPE", "kv_dtype", K::choice, T::engine, nullptr},
        {L"Speculative decoding", L"SPEC", "spec", K::choice, T::engine, nullptr},
        {L"Images (vision tower)", L"VISION", "vision", K::choice, T::engine, nullptr},
        {L"Requests at once", L"MAX_SEQS", "max_seqs", K::choice, T::engine, nullptr},
        {L"Context limit (tokens)", L"MAX_TOKENS", "max_tokens", K::choice, T::engine, nullptr},
        {L"VRAM budget (GiB)", L"VRAM_BUDGET_GIB", "vram_budget_gib", K::choice, T::engine, nullptr},
        {L"Load the engine", L"ENGINE_AUTOSTART", nullptr, K::choice, T::server, nullptr},
        {L"Unload when idle (seconds)", L"ENGINE_IDLE_UNLOAD_S", nullptr, K::edit, T::live, L"0: never"},
        {L"Smallest KV pool (tokens)", L"MIN_KV_TOKENS", nullptr, K::edit, T::server, L"Below this it does not load. 0: any size"},
    };
}

Row* row_of(const std::wstring& var) {
    for (auto& row : settings.rows)
        if (var == row.var) return &row;
    return nullptr;
}

std::wstring window_text(HWND window) {
    std::wstring text(GetWindowTextLengthW(window), L'\0');
    if (!text.empty()) GetWindowTextW(window, text.data(), static_cast<int>(text.size()) + 1);
    return text;
}

std::wstring value_of(const Row& row) {
    if (row.kind != Kind::choice) return trimmed(window_text(row.control));
    auto at = SendMessageW(row.control, CB_GETCURSEL, 0, 0);
    return at >= 0 && static_cast<size_t>(at) < row.values.size() ? row.values[at] : row.initial;
}

// Whether the value is the one the settings give now (a folder: by where it leads).
bool row_unchanged(const Row& row) {
    if (wcscmp(row.var, L"HF_HOME")) return value_of(row) == row.initial;
    wchar_t a[MAX_PATH * 4], b[MAX_PATH * 4];
    return GetFullPathNameW(value_of(row).c_str(), ARRAYSIZE(a), a, nullptr) && GetFullPathNameW(row.initial.c_str(), ARRAYSIZE(b), b, nullptr) &&
           _wcsicmp(a, b) == 0;
}

// Whether Apply writes it: another value, or one chosen that the file does not have yet.
bool row_changed(const Row& row) { return row.known && (!row_unchanged(row) || (row.chosen && !row.in_file)); }

std::wstring grouped(double number) {
    std::wstring digits = std::to_wstring(static_cast<long long>(number));
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) digits.insert(at, L",");
    return digits;
}

std::wstring plan_text(const Json& plan) {
    if (plan.type == Json::object && plan["error"].type == Json::string && plan["error"].text.find("VRAM") == std::string::npos) return L"no estimate";
    if (plan.type != Json::object || plan["error"].type == Json::string || !plan["fits"].flag) return L"does not fit now";
    return L"context " + grouped(plan["max_tokens"].value) + L", KV " + grouped(plan["kv_tokens"].value);
}

std::wstring value_name(const Row& row, const std::wstring& value) {
    if (value.empty() && !wcscmp(row.var, L"VRAM_BUDGET_GIB")) return L"automatic";
    return value;
}

// The list of a choice with its values; `value` stays selected and is added when the list does not have it.
void fill_choice(Row& row, std::vector<std::pair<std::wstring, std::wstring>> items, const std::wstring& value, bool enabled) {
    bool found = false;
    for (const auto& item : items) found = found || item.first == value;
    if (!found) items.insert(items.begin(), {value, value_name(row, value)});
    bool outer = settings.filling;  // (show_rows goes on after this)
    settings.filling = true;
    SendMessageW(row.control, CB_RESETCONTENT, 0, 0);
    row.values.clear();
    for (const auto& item : items) {
        row.values.push_back(item.first);
        SendMessageW(row.control, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(item.second.c_str()));
        if (item.first == value) SendMessageW(row.control, CB_SETCURSEL, row.values.size() - 1, 0);
    }
    EnableWindow(row.control, enabled);
    settings.filling = outer;
}

void set_message(const std::wstring& text, bool error, bool result) {
    settings.message_is_error = error;
    settings.message_is_result = result;
    SetWindowTextW(settings.message, text.c_str());
}

std::wstring engine_running_value(const Row& row) {
    const Json& running = settings.engine["running"];
    return env_text((running.type == Json::object ? running : settings.engine["settings"])[row.key]);
}

// Whether these weights can only come with a new server: another tokenizer, or not in the server's list.
bool weights_need_server(const std::wstring& value) {
    if (!settings.has_engine) return false;
    const Json& why = settings.engine["weights"][narrow(value)];
    return settings.engine["weights"].fields.empty() || why.type == Json::string ||
           std::none_of(settings.engine["weights"].fields.begin(), settings.engine["weights"].fields.end(),
                        [&](const auto& f) { return f.first == narrow(value); });
}

bool server_is_running(Shared* shared = nullptr) {
    std::lock_guard lock(app.mutex);
    if (shared) *shared = app.shared;
    return app.shared.phase == Phase::running;
}

bool engine_holds_vram() {
    std::lock_guard lock(app.mutex);
    const std::string& h = app.shared.health;
    return app.shared.phase == Phase::running && !h.empty() && h != "unloaded" && h != "stopped";
}

struct ApplyPlan {
    bool any = false, server = false, engine = false, live = false;
};

ApplyPlan apply_plan() {
    ApplyPlan plan;
    for (const auto& row : settings.rows) {
        if (!row_changed(row)) continue;
        plan.any = true;
        if (row_unchanged(row)) continue;  // (only written down: nothing to do)
        if (row.tier == Tier::server || (row.key && !strcmp(row.key, "weights") && weights_need_server(value_of(row))))
            plan.server = true;
        else if (row.tier == Tier::live)
            plan.live = true;
        else if (!settings.has_engine || value_of(row) != engine_running_value(row))
            plan.engine = true;
    }
    return plan;
}

std::wstring effect_text(const ApplyPlan& plan) {
    if (!plan.any) return L"";
    std::wstring text = L"Apply saves to settings.ini";
    bool running = server_is_running(), loaded = engine_holds_vram();
    const wchar_t* lost = L": requests in flight fail and the prefix cache is lost";
    if (!running) {
        bool active;
        Phase phase;
        {
            std::lock_guard lock(app.mutex);
            active = app.runner_active;
            phase = app.shared.phase;
        }
        if (settings.download_needed) text += L". The server then downloads what the folder does not have (20 GB or more)";
        return text + (!active && phase == Phase::failed ? L" and starts the server." : L". It is used when the server starts.");
    }
    if (plan.server) {
        text += L" and restarts the server";
        if (settings.download_needed) text += L", which downloads what the folder does not have (20 GB or more)";
        return text + (loaded ? lost : L"") + L".";
    }
    if (plan.engine) return text + (loaded ? std::wstring(L" and reloads the engine") + lost : std::wstring(L". The engine uses it when it loads")) + L".";
    return text + (plan.live ? L" and it applies at once." : L".");
}

bool is_number(const std::wstring& s) { return !s.empty() && std::all_of(s.begin(), s.end(), iswdigit); }

// What is wrong with the values ("": nothing).
std::wstring check_values() {
    std::wstring port = value_of(*row_of(L"PORT"));
    if (!is_number(port) || port.size() > 5 || _wtoi(port.c_str()) < 1 || _wtoi(port.c_str()) > 65535) return L"Port: a number from 1 to 65535.";
    wchar_t* end = nullptr;
    std::wstring idle = value_of(*row_of(L"ENGINE_IDLE_UNLOAD_S"));
    double seconds = wcstod(idle.c_str(), &end);
    if (idle.empty() || *end || seconds < 0) return L"Unload when idle: seconds, 0 or more.";
    if (!is_number(value_of(*row_of(L"MIN_KV_TOKENS")))) return L"Smallest KV pool: a number of tokens, 0 or more.";
    std::wstring host = value_of(*row_of(L"HOST"));
    if (host.empty() || host.find_first_of(L" \t\"") != std::wstring::npos) return L"Address: an IP address, such as 127.0.0.1.";
    if (value_of(*row_of(L"HF_HOME")).empty()) return L"Weights folder: a folder.";
    return {};
}

std::wstring full_path(const std::wstring& path) {
    wchar_t full[MAX_PATH * 4];  // (the working directory is the package's folder)
    return GetFullPathNameW(path.c_str(), ARRAYSIZE(full), full, nullptr) ? std::wstring(full) : path;
}

bool folder_changed() {
    const Row& row = *row_of(L"HF_HOME");
    return _wcsicmp(full_path(value_of(row)).c_str(), full_path(row.initial).c_str()) != 0;
}

// frontend/plan.py gives the estimates while the server cannot: it is stopped, uses another folder, or does not take
// the chosen weights (another tokenizer).
bool weights_from_scan() {
    const Row& weights = *row_of(L"WEIGHTS");
    return folder_changed() || !server_is_running() || (weights.known && weights_need_server(value_of(weights)));
}

// The note on the folder from the last scan.
void show_scan() {
    Row& weights = *row_of(L"WEIGHTS");
    std::wstring links;
    for (size_t i = 0; i < settings.scan_links.size() && i < 3; i++) links += (i ? L", " : L"") + settings.scan_links[i];
    if (settings.scan_links.size() > 3) links += L" and " + std::to_wstring(settings.scan_links.size() - 3) + L" more";
    if (!links.empty()) links = L"\nWindows cannot read these here (links made under WSL): " + links + L".";
    settings.folder_warning = !links.empty();
    ShowWindow(settings.fix_links, settings.scanned && !links.empty() ? SW_SHOW : SW_HIDE);
    if (!weights.known || !settings.scanned || !weights_from_scan()) {
        settings.download_needed = false;
        SetWindowTextW(settings.folder_note, settings.scanned && !links.empty() ? links.substr(1).c_str() : L"");
        return;
    }
    std::wstring current = value_of(weights);
    bool draft_needed = value_of(*row_of(L"SPEC")) != L"none" && !settings.scan_draft;
    bool found = std::find(settings.scan_weights.begin(), settings.scan_weights.end(), current) != settings.scan_weights.end();
    settings.download_needed = !found || draft_needed;
    std::wstring note;
    if (!exists(settings.scan_folder))
        note = L"This folder does not exist yet. ";
    if (!found)
        note += current + L" is not here: the server would download it.";
    else if (draft_needed)
        note += L"The DFlash2 draft is not here: the server would download it (or set Speculative decoding to none).";
    else
        note = std::to_wstring(settings.scan_weights.size()) + L" checkpoints of the model here.";
    settings.folder_warning = settings.folder_warning || settings.download_needed;
    SetWindowTextW(settings.folder_note, (note + links).c_str());
}

void update_controls() {
    if (!settings.window || settings.filling) return;
    show_scan();
    bool idle_zero = wcstod(value_of(*row_of(L"ENGINE_IDLE_UNLOAD_S")).c_str(), nullptr) == 0;
    bool on_demand = value_of(*row_of(L"ENGINE_AUTOSTART")) == L"0";
    SetWindowTextW(settings.load_warning, on_demand && idle_zero ? L"With both of these, requests get 503 until the engine is started on the stats page." : L"");
    std::wstring host = value_of(*row_of(L"HOST"));
    bool loopback = host == L"127.0.0.1" || host == L"localhost" || host == L"::1";
    SetWindowTextW(settings.host_warning, loopback ? L"" : L"The network: anyone there can use the API and change these settings (no authentication).");

    settings.keep.clear();
    std::wstring keep;
    if (settings.has_engine) {
        for (const auto& field : settings.engine["keep_env"].fields) {
            std::wstring var = widen(field.first), value = env_text(field.second);
            Row* row = row_of(var);
            if (!row || value_of(*row) == value) continue;
            settings.keep[var] = value;
            keep += (keep.empty() ? L"" : L", ") + var + L"=" + value_name(*row, value);
        }
    }
    SetWindowTextW(settings.keep_text, keep.empty() ? L"" : (L"Changed on the stats page, not in settings.ini: " + keep).c_str());
    ShowWindow(settings.keep_button, keep.empty() ? SW_HIDE : SW_SHOW);

    ApplyPlan plan = apply_plan();
    EnableWindow(settings.apply, plan.any);
    if (plan.any && !settings.message_is_result) {  // (a result stays until the next edit)
        std::wstring wrong = check_values();
        set_message(wrong.empty() ? effect_text(plan) : wrong, !wrong.empty(), false);
    } else if (!settings.message_is_result) {
        set_message(L"", false, false);
    }
    InvalidateRect(settings.window, nullptr, FALSE);
}

void load_rows() {
    std::vector<std::string> notes;
    auto env = server_env(&notes);
    auto file = read_settings();
    for (auto& row : settings.rows) {
        auto found = env.find(row.var);
        row.known = found != env.end();
        row.initial = row.known ? found->second : std::wstring();
        std::wstring var = row.var;
        row.in_file = file.count(var) != 0;
        row.chosen = false;
        if (var == L"HF_HOME") row.initial = file.count(var) ? file[var] : L"huggingface";
        if (var == L"ENGINE_AUTOSTART") row.initial = row.initial == L"0" ? L"0" : L"1";
        if (var == L"MIN_KV_TOKENS" && !row.known) row.known = true, row.initial = L"0";
        if (row.known && row.key && row.tier == Tier::engine && (var == L"MAX_SEQS" || var == L"MAX_TOKENS" || var == L"VRAM_BUDGET_GIB") && !row.initial.empty()) {
            wchar_t number[32];
            swprintf(number, 32, L"%g", wcstod(row.initial.c_str(), nullptr));  // (24.0 and 24 are the same choice)
            row.initial = number;
        }
    }
}

void show_rows() {
    settings.filling = true;
    for (auto& row : settings.rows) {
        std::wstring var = row.var;
        if (var == L"ENGINE_AUTOSTART") {
            fill_choice(row, {{L"0", L"At the first request"}, {L"1", L"When the server starts"}}, row.initial, true);
        } else if (row.kind == Kind::choice) {
            fill_choice(row, {{row.initial, row.known ? value_name(row, row.initial) : L"(the default)"}}, row.initial, false);
        } else {
            SetWindowTextW(row.control, row.initial.c_str());
        }
    }
    settings.filling = false;
}

// The output of a script of the package, with HF_HOME set to `folder` ("" if it did not run).
std::string python_output(const std::wstring& script, const std::wstring& args, const std::wstring& folder, const std::string& request = {}) {
    std::vector<std::string> notes;
    auto env = server_env(&notes);
    env[L"HF_HOME"] = folder;
    if (!request.empty()) env[L"Q27_PLAN_REQUEST"] = widen(request);
    std::wstring block = env_block(env);
    SECURITY_ATTRIBUTES inherit{sizeof inherit, nullptr, TRUE};
    HANDLE read_end, write_end;
    if (!CreatePipe(&read_end, &write_end, &inherit, 0)) return {};
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOW si{sizeof si};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = si.hStdError = nul;
    si.hStdOutput = write_end;
    PROCESS_INFORMATION pi{};
    std::wstring command = L"\"" + app.python + L"\" \"" + script + L"\"" + args;
    std::vector<wchar_t> line(command.begin(), command.end());
    line.push_back(L'\0');
    BOOL started = CreateProcessW(app.python.c_str(), line.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
                                  const_cast<wchar_t*>(block.c_str()), app.root.c_str(), &si, &pi);
    CloseHandle(write_end);
    CloseHandle(nul);
    std::string out;
    if (started) {
        char buffer[4096];
        DWORD n;
        while (ReadFile(read_end, buffer, sizeof buffer, &n, nullptr) && n) out.append(buffer, n);
        WaitForSingleObject(pi.hProcess, INFINITE);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
    CloseHandle(read_end);
    return out;
}

void post_result(SettingsResult* result) {
    if (!settings.window || !PostMessageW(settings.window, WM_APP_SETTINGS, 0, reinterpret_cast<LPARAM>(result))) delete result;
}

std::string settings_json();
void on_plan_answer(const Json& parsed);

// What the folder in the window has and the estimates for it, without the server (frontend/plan.py).
void scan_folder() {
    std::wstring folder = full_path(value_of(*row_of(L"HF_HOME")));
    settings.scanning = true;
    settings.plan_wanted = false;
    std::string request = "{\"settings\":" + settings_json();
    const Json& device = settings.engine["device"];
    if (settings.has_engine && device["free"].type == Json::number)  // (what the server's own plans assume)
        request += ",\"device\":[" + std::to_string(static_cast<long long>(device["free"].value)) + "," +
                   std::to_string(static_cast<long long>(device["total"].value)) + "]";
    request += "}";
    std::thread([folder, request] {
        auto* result = new SettingsResult{SettingsResult::scan};
        result->text = folder;
        result->answer.answered = true;
        result->answer.body = python_output(app.plan, L"", folder, request);
        post_result(result);
    }).detach();
}

void on_scan_answer(const SettingsResult& result) {
    settings.scanning = false;
    if (_wcsicmp(result.text.c_str(), full_path(value_of(*row_of(L"HF_HOME"))).c_str()) != 0) {  // (changed meanwhile)
        scan_folder();
        return;
    }
    Json parsed;
    if (!JsonReader(result.answer.body).read(&parsed) || parsed.type != Json::object) {
        SetWindowTextW(settings.summary, L"No estimate: the planner did not run. The log has the server's errors.");
        settings.scanned = true;
        return;
    }
    settings.scanned = true;
    settings.scan_folder = result.text;
    settings.scan_draft = parsed["draft"].flag;
    settings.scan_weights.clear();
    for (const auto& field : parsed["weights"].fields)
        if (field.second.type != Json::string) settings.scan_weights.push_back(widen(field.first));
    settings.scan_links.clear();
    for (const auto& w : parsed["links"].items) settings.scan_links.push_back(widen(w.text));
    if (weights_from_scan())
        on_plan_answer(parsed);
    else
        settings.plan_wanted = true;  // (the server's answer)
}

// The folder chosen in Explorer's dialog ("": cancelled).
std::wstring pick_folder(HWND owner, const std::wstring& start) {
    std::wstring chosen;
    if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE))) return chosen;
    IFileOpenDialog* dialog = nullptr;
    if (SUCCEEDED(CoCreateInstance(__uuidof(FileOpenDialog), nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) {
        DWORD options = 0;
        dialog->GetOptions(&options);
        dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
        dialog->SetTitle(L"Weights folder (a Hugging Face cache)");
        IShellItem* item = nullptr;
        if (SUCCEEDED(SHCreateItemFromParsingName(start.c_str(), nullptr, IID_PPV_ARGS(&item)))) {
            dialog->SetFolder(item);
            item->Release();
        }
        if (SUCCEEDED(dialog->Show(owner)) && SUCCEEDED(dialog->GetResult(&item))) {
            PWSTR path = nullptr;
            if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
                chosen = path;
                CoTaskMemFree(path);
            }
            item->Release();
        }
        dialog->Release();
    }
    CoUninitialize();
    return chosen;
}

void browse_folder() {
    Row& row = *row_of(L"HF_HOME");
    std::wstring folder = pick_folder(settings.window, full_path(value_of(row)));
    if (folder.empty()) return;
    // (HF_HOME is the folder above hub\, easy to choose by mistake: the models are in hub\ under it)
    size_t slash = folder.find_last_of(L'\\');
    if (slash != std::wstring::npos && slash > 2 && _wcsicmp(folder.c_str() + slash + 1, L"hub") == 0) {
        WIN32_FIND_DATAW found;
        HANDLE models = FindFirstFileW((folder + L"\\models--*").c_str(), &found);
        if (models != INVALID_HANDLE_VALUE) {
            FindClose(models);
            folder.resize(slash);
        }
    }
    SetWindowTextW(row.control, folder.c_str());  // (EN_CHANGE: the weights follow)
}

// The links Windows cannot read in the folder, as hard links to the same files (download_weights.py --fix-links).
void fix_links() {
    std::wstring folder = full_path(value_of(*row_of(L"HF_HOME")));
    std::wstring question = L"Replace the links in " + folder + L" with hard links to the same files?\n\n"
                            L"Windows and WSL then both read them. Nothing is copied or downloaded.";
    if (MessageBoxW(settings.window, question.c_str(), APP_TITLE, MB_OKCANCEL | MB_ICONQUESTION) != IDOK) return;
    EnableWindow(settings.fix_links, FALSE);
    set_message(L"Replacing the links...", false, true);
    std::thread([folder] {
        auto* result = new SettingsResult{SettingsResult::fixed};
        result->answer.body = python_output(app.download, L" --fix-links", folder);
        post_result(result);
    }).detach();
}

void on_links_fixed(const std::string& body) {
    Json parsed;
    EnableWindow(settings.fix_links, TRUE);
    if (!JsonReader(body).read(&parsed) || parsed["fixed"].type != Json::number) {
        set_message(L"The links could not be replaced.", true, true);
    } else {
        size_t failed = parsed["failed"].items.size();
        for (const auto& item : parsed["failed"].items) tray_log("--fix-links: " + item.text);
        std::wstring text = L"Replaced " + std::to_wstring(static_cast<long long>(parsed["fixed"].value)) + L" links.";
        if (failed) text += L" " + std::to_wstring(failed) + L" failed: the log has them.";
        set_message(text, failed != 0, true);
    }
    settings.scanned = false;      // (the next tick asks the folder and the server again)
    settings.has_engine = false;
}

void fetch_engine() {
    settings.busy = true;
    int port = app.port;
    std::thread([port] {
        auto* result = new SettingsResult{SettingsResult::engine};
        result->answer = http_call(port, L"GET", L"/admin/engine", "");
        post_result(result);
    }).detach();
}

// The start settings as the controls have them, for the admin API.
std::string settings_json() {
    std::string out;
    for (const auto& row : settings.rows) {
        if (!row.key || !row.known) continue;
        std::wstring value = value_of(row);
        std::wstring var = row.var;
        std::string text;
        if (var == L"WEIGHTS" && !weights_from_scan() && weights_need_server(value)) continue;  // (the server does not take them)
        if (var == L"MAX_SEQS" || var == L"MAX_TOKENS") text = narrow(value);
        else if (var == L"VRAM_BUDGET_GIB") text = value.empty() ? "null" : narrow(value);
        else text = json_quote(value);
        out += (out.empty() ? "\"" : ",\"") + std::string(row.key) + "\":" + text;
    }
    return "{" + out + "}";
}

void fetch_plan() {
    settings.busy = true;
    settings.plan_wanted = false;
    int port = app.port;
    std::string body = "{\"settings\":" + settings_json() + "}";
    std::thread([port, body] {
        auto* result = new SettingsResult{SettingsResult::plan};
        result->answer = http_call(port, L"POST", L"/admin/engine/plan", body);
        post_result(result);
    }).detach();
}

// What neither the file nor the environment has: the server's default.
void fill_defaults(const Json& at_start) {
    if (at_start.type != Json::object) return;
    for (auto& row : settings.rows) {
        if (row.known || !row.key) continue;
        row.known = true;
        row.initial = env_text(at_start[row.key]);
        fill_choice(row, {{row.initial, value_name(row, row.initial)}}, row.initial, false);
    }
}

void on_engine_answer(const HttpAnswer& answer) {
    Json parsed;
    if (!answer.answered || answer.status != 200 || !JsonReader(answer.body).read(&parsed)) return;
    settings.engine = std::move(parsed);
    settings.has_engine = true;
    if (settings.restarting) set_message(L"Saved. The server has restarted.", false, true);
    settings.restarting = false;
    fill_defaults(settings.engine["at_start"]);
    settings.plan_wanted = true;
}

// The answer of POST /admin/engine/plan or frontend/plan.py: the choices and what each leaves for the context.
void on_plan_answer(const Json& parsed) {
    fill_defaults(parsed["at_start"]);
    const Json& plan = parsed["plan"];
    if (plan.type != Json::object) {
        SetWindowTextW(settings.summary, (L"No estimate: " + widen(parsed["error"].text)).c_str());
        return;
    }
    const Json& device = parsed["device"];
    constexpr double GIB = 1 << 30;
    wchar_t pools[160] = L"";
    if (plan["fits"].type == Json::boolean) {
        swprintf(pools, 160, L"\nDevice pools %.1f of %.1f GiB budget; %.1f GiB free of %.1f GiB.", plan["device"].value / GIB,
                 plan["budget"].value / GIB, device["free"].value / GIB, device["total"].value / GIB);
    }
    std::wstring error = widen(plan["error"].text);
    std::wstring text = !error.empty()     ? (error.find(L"VRAM") != std::wstring::npos ? L"Does not fit: " : L"No estimate: ") + error
                        : !plan["fits"].flag ? std::wstring(L"These do not fit the VRAM free now.") + pools
                                             : L"Longest context " + grouped(plan["max_tokens"].value) + L" tokens; KV pool " +
                                                   grouped(plan["kv_tokens"].value) + L" tokens, shared by the requests." + pools;
    SetWindowTextW(settings.summary, text.c_str());

    for (auto& row : settings.rows) {
        if (!row.key) continue;
        if (SendMessageW(row.control, CB_GETDROPPEDSTATE, 0, 0)) continue;  // (not while someone is choosing)
        std::vector<std::pair<std::wstring, std::wstring>> items;
        std::wstring current = value_of(row);
        for (const auto& item : parsed["variants"][row.key].items) {
            std::wstring value = env_text(item["value"]);
            std::wstring var = row.var;
            std::wstring label = value_name(row, value) + L"  -  " + plan_text(item["plan"]);
            if (var == L"WEIGHTS") {
                const std::string name = narrow(value);
                const Json& why = parsed["weights"][name];
                const Json& server_why = settings.engine["weights"][name];
                bool tokenizer = server_why.text.find("tokenizer") != std::string::npos || why.text.find("tokenizer") != std::string::npos;
                if (why.type == Json::string && !tokenizer) {
                    if (value != current) continue;  // (not downloaded)
                    label = value + L"  (not in this folder)";
                }
                if (tokenizer) label += L"  (another tokenizer: restarts the server)";
            }
            items.push_back({value, label});
        }
        if (!strcmp(row.key, "weights") && std::none_of(items.begin(), items.end(), [&](const auto& item) { return item.first == current; }))
            items.insert(items.begin(), {current, current + L"  (not in this folder)"});
        fill_choice(row, std::move(items), current, true);
    }
}

void on_plan_http_answer(const HttpAnswer& answer) {
    Json parsed;
    if (!answer.answered || answer.status != 200 || !JsonReader(answer.body).read(&parsed)) {
        SetWindowTextW(settings.summary, (L"No estimate: " + api_error(answer)).c_str());
        return;
    }
    on_plan_answer(parsed);
}

// Writes the changed settings in the [env] section; the other lines stay as they are.
bool write_settings(const std::vector<std::pair<std::wstring, std::wstring>>& changes) {
    std::vector<std::wstring> lines = lines_of(read_text(app.ini));
    size_t section = std::find_if(lines.begin(), lines.end(), is_env_section) - lines.begin();
    if (section == lines.size()) lines.push_back(L"[env]");
    for (const auto& [name, value] : changes) {
        bool done = false;
        for (size_t at = section + 1; at < lines.size() && trimmed(lines[at]).rfind(L'[', 0) != 0;) {
            if (name_of(lines[at]) != upper(name)) {
                at++;
            } else if (done) {
                lines.erase(lines.begin() + at);  // (the last of several wins when reading)
            } else {
                lines[at++] = name + L"=" + value;
                done = true;
            }
        }
        if (!done) lines.insert(lines.begin() + section + 1, name + L"=" + value);
    }
    std::wstring text;
    for (const auto& line : lines) text += line + L"\r\n";
    std::string utf8 = narrow(text);
    std::wstring temporary = app.ini + L".tmp";
    HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    DWORD written = 0;
    BOOL ok = WriteFile(file, utf8.data(), static_cast<DWORD>(utf8.size()), &written, nullptr) && written == utf8.size();
    CloseHandle(file);
    return ok && MoveFileExW(temporary.c_str(), app.ini.c_str(), MOVEFILE_REPLACE_EXISTING);
}

void apply_settings() {
    std::wstring wrong = check_values();
    if (!wrong.empty()) {
        set_message(wrong, true, false);
        return;
    }
    ApplyPlan plan = apply_plan();
    if (!plan.any) return;
    bool loaded = engine_holds_vram();
    if ((loaded && (plan.server || plan.engine)) || (settings.download_needed && (plan.server || !server_is_running()))) {
        std::wstring question = effect_text(plan) + L"\n\nContinue?";
        if (MessageBoxW(settings.window, question.c_str(), APP_TITLE, MB_OKCANCEL | MB_ICONWARNING) != IDOK) return;
    }
    std::vector<std::pair<std::wstring, std::wstring>> changes;
    for (const auto& row : settings.rows)
        if (row_changed(row)) changes.push_back({row.var, value_of(row)});
    if (!write_settings(changes)) {
        set_message(L"Cannot write " + app.ini + L". Nothing changed.", true, true);
        return;
    }
    for (const auto& [name, value] : changes) tray_log("settings.ini: " + narrow(name + L"=" + value));

    std::string engine_body = "{\"action\":\"restart\",\"settings\":" + settings_json() + "}";
    std::string idle = narrow(value_of(*row_of(L"ENGINE_IDLE_UNLOAD_S")));
    for (auto& row : settings.rows) {
        if (row_changed(row)) row.in_file = true;
        row.chosen = false;
        row.initial = value_of(row);
    }

    bool active;
    Phase phase;
    {
        std::lock_guard lock(app.mutex);
        active = app.runner_active;
        phase = app.shared.phase;
    }
    settings.has_engine = false;  // (asked again below or when the server is up)
    if (plan.server && active) {
        app.restart_pending = settings.restarting = true;
        stop_server();
        set_message(L"Saved. The server is restarting.", false, true);
    } else if (!active) {
        if (phase == Phase::failed) start_server();
        set_message(phase == Phase::failed ? L"Saved. The server is starting." : L"Saved. It is used when the server starts.", false, true);
    } else if (!server_is_running()) {
        set_message(L"Saved. It is used when the server starts: restart it to use it now.", false, true);
    } else if (plan.engine || plan.live) {
        settings.busy = true;
        set_message(L"Saved. Applying...", false, true);
        int port = app.port;
        bool engine = plan.engine, live = plan.live;
        std::thread([port, engine_body, idle, engine, live] {
            auto* result = new SettingsResult{SettingsResult::applied};
            result->text = L"Saved and applied.";
            if (live) {
                HttpAnswer a = http_call(port, L"POST", L"/admin/engine/idle", "{\"idle_unload_s\":" + idle + "}");
                if (a.status != 200) {
                    result->failed = true;
                    result->text = L"Saved in settings.ini. Not applied: " + api_error(a);
                }
            }
            if (engine && !result->failed) {
                HttpAnswer state = http_call(port, L"GET", L"/admin/engine", "");
                Json parsed;
                JsonReader(state.body).read(&parsed);
                std::string body = engine_body;
                bool stopped = parsed["state"].text == "stopped";
                if (stopped) body.replace(body.find("restart"), 7, "save");  // (nothing to reload: for the next load)
                HttpAnswer a = http_call(port, L"POST", L"/admin/engine", body);
                if (a.status != 200) {
                    result->failed = true;
                    result->text = L"Saved in settings.ini. Not applied: " + api_error(a);
                } else {
                    result->text = stopped ? L"Saved. The engine uses it when it loads." : L"Saved. The engine is reloading.";
                }
            }
            post_result(result);
        }).detach();
    } else {
        set_message(L"Saved.", false, true);
    }
    update_controls();
}

void open_ini() {
    if (!exists(app.ini)) write_settings({});
    ShellExecuteW(settings.window, L"open", L"notepad.exe", (L"\"" + app.ini + L"\"").c_str(), nullptr, SW_SHOWNORMAL);
}

void set_settings_font() {
    UINT dpi = GetDpiForWindow(settings.window);
    NONCLIENTMETRICSW metrics{sizeof metrics};
    SystemParametersInfoForDpi(SPI_GETNONCLIENTMETRICS, sizeof metrics, &metrics, 0, dpi);
    HFONT font = CreateFontIndirectW(&metrics.lfMessageFont);
    EnumChildWindows(settings.window, [](HWND child, LPARAM f) {
        SendMessageW(child, WM_SETFONT, static_cast<WPARAM>(f), TRUE);
        return TRUE;
    }, reinterpret_cast<LPARAM>(font));
    if (settings.font) DeleteObject(settings.font);
    settings.font = font;
}

void layout_settings() {
    UINT dpi = GetDpiForWindow(settings.window);
    auto px = [dpi](int dip) { return MulDiv(dip, dpi, 96); };
    constexpr int WIDTH = 680, MARGIN = 12, INSET = 12, ROW = 28, LABEL = 180;
    const int control_x = MARGIN + INSET + LABEL, control_w = WIDTH - control_x - MARGIN - INSET;
    int y = MARGIN;
    auto place = [&](HWND w, int x, int top, int width, int height) { MoveWindow(w, px(x), px(top), px(width), px(height), TRUE); };
    place(settings.state, MARGIN, y, WIDTH - 2 * MARGIN, 20);
    y += 26;

    auto place_rows = [&](size_t first, size_t last) {
        for (size_t i = first; i <= last; i++) {
            Row& row = settings.rows[i];
            place(row.label_window, MARGIN + INSET, y + 3, LABEL - 6, 20);
            int width = row.hint ? 110 : control_w;
            if (!wcscmp(row.var, L"HF_HOME")) {
                width = control_w - 100;
                place(settings.browse, control_x + width + 8, y - 1, 92, 26);
            }
            place(row.control, control_x, y, width, row.kind == Kind::edit ? 23 : 240);
            if (row.hint) place(row.hint_window, control_x + width + 8, y + 3, control_w - width - 8, 20);
            y += ROW;
        }
    };
    auto group = [&](int index, int top) { place(settings.groups[index], MARGIN, top, WIDTH - 2 * MARGIN, y + 8 - top); y += 16; };

    int top = y;
    y += 20;
    place_rows(0, ROW_SERVER_LAST);
    place(settings.folder_note, MARGIN + INSET, y, WIDTH - 2 * (MARGIN + INSET) - 110, 52);
    place(settings.fix_links, WIDTH - MARGIN - INSET - 100, y, 100, 26);
    y += 54;
    place(settings.host_warning, MARGIN + INSET, y, WIDTH - 2 * (MARGIN + INSET), 20);
    y += 22;
    group(2, top);

    top = y;
    y += 20;
    place_rows(ROW_SERVER_LAST + 1, ROW_ENGINE_LAST);
    place(settings.summary, MARGIN + INSET, y + 2, WIDTH - 2 * (MARGIN + INSET), 36);
    y += 40;
    group(0, top);

    top = y;
    y += 20;
    place_rows(ROW_ENGINE_LAST + 1, ROW_LOAD_LAST);
    place(settings.load_warning, MARGIN + INSET, y, WIDTH - 2 * (MARGIN + INSET), 20);
    y += 22;
    group(1, top);

    place(settings.keep_text, MARGIN, y + 3, WIDTH - 2 * MARGIN - 110, 36);
    place(settings.keep_button, WIDTH - MARGIN - 100, y, 100, 26);
    y += 40;
    place(settings.message, MARGIN, y, WIDTH - 2 * MARGIN, 36);
    y += 40;
    place(settings.open_ini, MARGIN, y, 130, 28);
    place(settings.apply, WIDTH - MARGIN - 190, y, 90, 28);
    place(settings.close, WIDTH - MARGIN - 90, y, 90, 28);
    y += 28 + MARGIN;

    RECT frame{0, 0, px(WIDTH), px(y)};
    DWORD style = GetWindowLongW(settings.window, GWL_STYLE);
    AdjustWindowRectExForDpi(&frame, style, FALSE, GetWindowLongW(settings.window, GWL_EXSTYLE), dpi);
    SetWindowPos(settings.window, nullptr, 0, 0, frame.right - frame.left, frame.bottom - frame.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
}

void create_settings_controls(HWND window) {
    HINSTANCE instance = GetModuleHandleW(nullptr);
    auto child = [&](const wchar_t* type, const wchar_t* text, DWORD style, int id = 0, DWORD ex = 0) {
        return CreateWindowExW(ex, type, text, WS_CHILD | WS_VISIBLE | style, 0, 0, 0, 0, window,
                               reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), instance, nullptr);
    };
    settings.state = child(L"STATIC", L"", SS_NOPREFIX);
    settings.groups[2] = child(L"BUTTON", L"Server: changing these restarts it", BS_GROUPBOX);
    settings.groups[0] = child(L"BUTTON", L"Engine: what it loads", BS_GROUPBOX);
    settings.groups[1] = child(L"BUTTON", L"When it holds the VRAM", BS_GROUPBOX);
    settings.rows = make_rows();
    int id = IDC_ROW;
    for (auto& row : settings.rows) {
        row.label_window = child(L"STATIC", row.label, SS_NOPREFIX);
        if (row.kind == Kind::choice)
            row.control = child(L"COMBOBOX", nullptr, WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST, id);
        else if (row.kind == Kind::combo_edit)
            row.control = child(L"COMBOBOX", nullptr, WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWN | CBS_AUTOHSCROLL, id);
        else
            row.control = child(L"EDIT", nullptr, WS_TABSTOP | ES_AUTOHSCROLL, id, WS_EX_CLIENTEDGE);
        if (row.hint) row.hint_window = child(L"STATIC", row.hint, SS_NOPREFIX);
        if (!wcscmp(row.var, L"HF_HOME")) settings.browse = child(L"BUTTON", L"&Browse...", WS_TABSTOP, IDC_BROWSE);
        id++;
    }
    SendMessageW(row_of(L"HOST")->control, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"127.0.0.1"));
    SendMessageW(row_of(L"HOST")->control, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"0.0.0.0"));
    settings.summary = child(L"STATIC", L"Reading the weights folder...", SS_NOPREFIX);
    settings.load_warning = child(L"STATIC", L"", SS_NOPREFIX);
    settings.host_warning = child(L"STATIC", L"", SS_NOPREFIX);
    settings.folder_note = child(L"STATIC", L"", SS_NOPREFIX);
    settings.fix_links = child(L"BUTTON", L"Fix the links", WS_TABSTOP, IDC_FIX_LINKS);
    ShowWindow(settings.fix_links, SW_HIDE);
    settings.keep_text = child(L"STATIC", L"", SS_NOPREFIX);
    settings.keep_button = child(L"BUTTON", L"Use these", WS_TABSTOP, IDC_USE_KEEP);
    ShowWindow(settings.keep_button, SW_HIDE);
    settings.message = child(L"STATIC", L"", SS_NOPREFIX);
    settings.open_ini = child(L"BUTTON", L"Open settings.ini", WS_TABSTOP, IDC_OPEN_INI);
    settings.apply = child(L"BUTTON", L"&Apply", WS_TABSTOP | BS_DEFPUSHBUTTON, IDC_APPLY);
    settings.close = child(L"BUTTON", L"Close", WS_TABSTOP, IDC_CLOSE);
}

// Once a second while the window is open: the state, and the server's answers when they are missing or out of date.
void settings_tick() {
    Shared shared;
    bool running = server_is_running(&shared);
    SetWindowTextW(settings.state, view_of(shared).text.c_str());
    if (!running && settings.has_engine) settings.has_engine = false;
    if (!running && !settings.scanned) {
        for (auto& row : settings.rows)
            if (row.key) EnableWindow(row.control, FALSE);
    }
    if (running && !settings.has_engine && !settings.busy) fetch_engine();
    if (!settings.scanning && (!settings.scanned || (settings.plan_wanted && weights_from_scan())))
        scan_folder();
    else if (running && settings.has_engine && !settings.busy && settings.plan_wanted && !weights_from_scan())
        fetch_plan();
}

LRESULT CALLBACK settings_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    switch (message) {
        case WM_CREATE:
            settings.window = window;
            create_settings_controls(window);
            set_settings_font();
            load_rows();
            show_rows();
            layout_settings();
            update_controls();
            settings_tick();
            SetTimer(window, TIMER_TICK, 1000, nullptr);
            return 0;
        case WM_TIMER:
            if (wparam == TIMER_PLAN) {
                KillTimer(window, TIMER_PLAN);
                settings.plan_wanted = true;
                update_controls();
            } else if (wparam == TIMER_SCAN) {
                KillTimer(window, TIMER_SCAN);
                settings.scanned = false;
                settings.plan_wanted = true;  // (the server's list again, if back to its folder)
                if (!settings.scanning) scan_folder();
                update_controls();
            }
            settings_tick();
            return 0;
        case WM_APP_SETTINGS: {
            std::unique_ptr<SettingsResult> result(reinterpret_cast<SettingsResult*>(lparam));
            if (result->what == SettingsResult::scan) {
                on_scan_answer(*result);
                update_controls();
                if (settings.plan_wanted) settings_tick();
                return 0;
            }
            if (result->what == SettingsResult::fixed) {
                on_links_fixed(result->answer.body);
                update_controls();
                return 0;
            }
            settings.busy = false;
            if (result->what == SettingsResult::engine) {
                on_engine_answer(result->answer);
            } else if (result->what == SettingsResult::plan) {
                on_plan_http_answer(result->answer);
            } else {
                set_message(result->text, result->failed, true);
                settings.has_engine = false;
            }
            update_controls();
            if (settings.has_engine && settings.plan_wanted) settings_tick();  // (otherwise the next tick asks again)
            return 0;
        }
        case WM_COMMAND: {
            int id = LOWORD(wparam), code = HIWORD(wparam);
            if (id == IDC_APPLY || id == IDOK) {
                apply_settings();
            } else if (id == IDC_CLOSE || id == IDCANCEL) {
                DestroyWindow(window);
            } else if (id == IDC_OPEN_INI) {
                open_ini();
            } else if (id == IDC_FIX_LINKS) {
                fix_links();
            } else if (id == IDC_BROWSE) {
                browse_folder();
            } else if (id == IDC_USE_KEEP) {
                for (const auto& [var, value] : std::map(settings.keep)) {
                    Row* row = row_of(var);
                    if (row->kind == Kind::choice)
                        fill_choice(*row, {{value, value_name(*row, value)}}, value, true);
                    else
                        SetWindowTextW(row->control, value.c_str());
                }
                settings.plan_wanted = true;
                update_controls();
            } else if (id >= IDC_ROW && !settings.filling && (code == CBN_SELCHANGE || code == CBN_SELENDOK || code == EN_CHANGE || code == CBN_EDITCHANGE)) {
                settings.rows[id - IDC_ROW].chosen = true;  // (CBN_SELENDOK: also the item that was selected already)
                settings.message_is_result = false;
                // (at CBN_SELCHANGE an address's edit box still has the old text, so also a moment later)
                int row = id - IDC_ROW;
                if ((row > ROW_SERVER_LAST && row <= ROW_ENGINE_LAST) || code == CBN_SELCHANGE || code == CBN_SELENDOK) SetTimer(window, TIMER_PLAN, 100, nullptr);
                if (row == ROW_SERVER_LAST) SetTimer(window, TIMER_SCAN, 500, nullptr);
                update_controls();
            }
            return 0;
        }
        case WM_CTLCOLORSTATIC: {
            auto control = reinterpret_cast<HWND>(lparam);
            HDC dc = reinterpret_cast<HDC>(wparam);
            bool warn = control == settings.load_warning || control == settings.host_warning ||
                        (control == settings.folder_note && settings.folder_warning) ||
                        (control == settings.message && settings.message_is_error);
            SetTextColor(dc, warn ? RGB(192, 0, 0) : GetSysColor(COLOR_WINDOWTEXT));
            SetBkColor(dc, GetSysColor(COLOR_BTNFACE));
            return reinterpret_cast<LRESULT>(GetSysColorBrush(COLOR_BTNFACE));
        }
        case WM_DPICHANGED: {
            auto* r = reinterpret_cast<RECT*>(lparam);
            SetWindowPos(window, nullptr, r->left, r->top, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
            set_settings_font();
            layout_settings();
            return 0;
        }
        case WM_DESTROY: {
            KillTimer(window, TIMER_TICK);
            KillTimer(window, TIMER_PLAN);
            KillTimer(window, TIMER_SCAN);
            if (settings.font) DeleteObject(settings.font);
            settings = SettingsWindow{};
            return 0;
        }
    }
    return DefWindowProcW(window, message, wparam, lparam);
}

void show_settings() {
    if (!settings.window) {
        std::wstring title = std::wstring(APP_TITLE) + L" settings";
        CreateWindowExW(WS_EX_CONTROLPARENT, SETTINGS_CLASS, title.c_str(), WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                        CW_USEDEFAULT, CW_USEDEFAULT, 700, 800, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
    }
    ShowWindow(settings.window, IsIconic(settings.window) ? SW_RESTORE : SW_SHOW);
    SetForegroundWindow(settings.window);
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
    AppendMenuW(menu, MF_STRING, ID_SETTINGS, L"S&ettings...");
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
            if (app.quitting) {
                DestroyWindow(window);
            } else {
                if (app.restart_pending) start_server();
                app.restart_pending = false;
                refresh();
            }
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
                case ID_SETTINGS: show_settings(); break;
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
            if (settings.window) DestroyWindow(settings.window);
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
    app.plan = app.root + L"\\app\\frontend\\plan.py";
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
    WNDCLASSEXW settings_class = log_class;
    settings_class.lpfnWndProc = settings_proc;
    settings_class.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
    settings_class.lpszClassName = SETTINGS_CLASS;
    RegisterClassExW(&settings_class);

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
        if (settings.window && IsDialogMessageW(settings.window, &message)) continue;  // (Tab, Enter, Esc)
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    CloseHandle(mutex);
    return 0;
}
