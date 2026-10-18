//! Hardcoded shell encyclopedia: a curated, verified reference of Windows
//! PowerShell commands for controlling and inspecting the whole PC. Exposed to
//! the model as the `shell_help` tool so it can look up the EXACT command for a
//! task (logs, processes, network, services, users, disk, …) instead of guessing
//! cmdlet/flag combinations. run_shell executes via Windows PowerShell 5.1.

use async_trait::async_trait;
use qo_core::{Tool, ToolContext, ToolResult};
use serde_json::{json, Value};

use crate::util::{js_string, js_truthy};

#[derive(Clone, Copy, Debug)]
pub struct ShellCmd {
    pub cmd: &'static str,
    pub desc: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct ShellTopic {
    pub topic: &'static str,
    pub keywords: &'static [&'static str],
    pub note: Option<&'static str>,
    pub commands: &'static [ShellCmd],
}

const fn c(cmd: &'static str, desc: &'static str) -> ShellCmd {
    ShellCmd { cmd, desc }
}

pub const SHELL_ENCYCLOPEDIA: &[ShellTopic] = &[
    ShellTopic {
        topic: "login & logon history (event logs)",
        keywords: &["login", "logon", "logoff", "signin", "sign in", "who logged", "event log", "winevent", "security log", "failed login", "4624", "4625", "auth", "logins"],
        note: Some("Reading the Security log requires an elevated (Administrator) shell. Key IDs: 4624 successful logon, 4625 failed logon, 4634 logoff, 4647 user-initiated logoff."),
        commands: &[
            c(r"Get-WinEvent -FilterHashtable @{LogName='Security'; Id=4624; StartTime=(Get-Date).AddDays(-7)} | Select-Object TimeCreated, @{N='User';E={$_.Properties[5].Value}}, @{N='LogonType';E={$_.Properties[8].Value}} | Format-Table -Auto", "Successful logons in the last 7 days (with user + logon type)"),
            c(r"Get-WinEvent -FilterHashtable @{LogName='Security'; Id=4625; StartTime=(Get-Date).AddDays(-7)} | Select-Object TimeCreated, @{N='User';E={$_.Properties[5].Value}} | Format-Table -Auto", "Failed logon attempts in the last 7 days"),
            c(r"Get-WinEvent -FilterHashtable @{LogName='Security'; Id=4624,4634; StartTime=(Get-Date).AddDays(-1)} | Sort-Object TimeCreated | Format-Table TimeCreated, Id -Auto", "Logon/logoff timeline for the last day"),
            c(r"Get-WinEvent -LogName Security -MaxEvents 50 | Where-Object Id -in 4624,4625 | Format-Table TimeCreated, Id, Message -Wrap", "Most recent 50 security logon events (no admin filter syntax)"),
            c("quser", "Currently logged-on interactive sessions"),
            c(r"Get-WinEvent -ListLog * | Where-Object RecordCount -gt 0 | Sort-Object RecordCount -Descending | Select-Object LogName, RecordCount -First 20", "Which event logs have entries"),
        ],
    },
    ShellTopic {
        topic: "system information",
        keywords: &["system", "os", "version", "computer info", "build", "spec", "machine", "windows version", "uptime", "boot"],
        note: None,
        commands: &[
            c("Get-ComputerInfo | Select-Object CsName, OsName, OsVersion, OsBuildNumber, WindowsProductName, CsManufacturer, CsModel", "Concise machine + OS summary"),
            c("[Environment]::OSVersion.Version; (Get-CimInstance Win32_OperatingSystem).Caption", "OS version + edition"),
            c("(Get-CimInstance Win32_OperatingSystem).LastBootUpTime", "Last boot time (uptime)"),
            c("Get-HotFix | Sort-Object InstalledOn -Descending | Select-Object -First 15", "Recently installed updates/patches"),
        ],
    },
    ShellTopic {
        topic: "processes",
        keywords: &["process", "processes", "task", "tasklist", "kill", "cpu usage", "memory usage", "running app", "pid"],
        note: None,
        commands: &[
            c(r"Get-Process | Sort-Object CPU -Descending | Select-Object -First 15 Name, Id, CPU, @{N='MB';E={[int]($_.WorkingSet64/1MB)}}", "Top processes by CPU with memory (MB)"),
            c("Get-Process -Name <name> | Format-Table Name, Id, CPU, WorkingSet -Auto", "Find a process by name"),
            c("Stop-Process -Name <name> -Force", "Kill processes by name (Remove-Item-class: confirm first)"),
            c("Stop-Process -Id <pid> -Force", "Kill one process by PID"),
        ],
    },
    ShellTopic {
        topic: "services",
        keywords: &["service", "services", "daemon", "start service", "stop service", "windows service"],
        note: None,
        commands: &[
            c("Get-Service | Where-Object Status -eq 'Running' | Sort-Object DisplayName | Format-Table Name, DisplayName -Auto", "All running services"),
            c("Get-Service -Name <name>", "Status of one service"),
            c("Start-Service -Name <name>; Stop-Service -Name <name>; Restart-Service -Name <name>", "Control a service (needs admin)"),
            c(r"Get-Service | Where-Object {$_.StartType -eq 'Automatic' -and $_.Status -ne 'Running'}", "Auto-start services that aren't running"),
        ],
    },
    ShellTopic {
        topic: "network & connectivity",
        keywords: &["network", "ip", "ipconfig", "wifi", "wlan", "dns", "ping", "port", "tcp", "netstat", "connection", "adapter", "internet", "ssid"],
        note: None,
        commands: &[
            c("Get-NetIPConfiguration | Format-List InterfaceAlias, IPv4Address, IPv4DefaultGateway, DnsServer", "IP / gateway / DNS per interface"),
            c("ipconfig /all", "Full IP configuration (classic)"),
            c("netsh wlan show interfaces", "Current Wi-Fi SSID, signal, state"),
            c("Test-NetConnection -ComputerName <host> -Port <port>", "Test reachability of a host:port"),
            c("Get-NetTCPConnection -State Established | Select-Object LocalAddress, LocalPort, RemoteAddress, RemotePort, OwningProcess | Format-Table -Auto", "Active TCP connections + owning PID"),
            c("Resolve-DnsName <hostname>", "DNS lookup"),
            c("Get-NetAdapter | Format-Table Name, Status, LinkSpeed, MacAddress -Auto", "Network adapters"),
        ],
    },
    ShellTopic {
        topic: "internet / bandwidth speed",
        keywords: &["speed", "internet speed", "download speed", "upload speed", "bandwidth", "mbps", "speedtest", "speed test", "how fast", "connection speed", "throughput", "fast is my"],
        note: None,
        commands: &[
            c(r#"$b=10000000; $r=[Net.HttpWebRequest]::Create("https://speed.cloudflare.com/__down?bytes=$b"); $r.Timeout=15000; $r.ReadWriteTimeout=15000; try { $t=Measure-Command { $resp=$r.GetResponse(); $st=$resp.GetResponseStream(); $bu=New-Object byte[] 65536; $n=0; while(($k=$st.Read($bu,0,65536)) -gt 0){$n+=$k}; $resp.Close() }; "{0:N1} Mbps down" -f ($n*8/$t.TotalSeconds/1e6) } catch { "download failed (host blocked or no internet): " + $_.Exception.Message }"#, "DOWNLOAD speed — ~10MB sample, streamed with a 15s timeout so it fails fast instead of hanging if the host is blocked. Reports Mbps, or a clear error to fall back on latency/link-speed below"),
            c("(Test-Connection 1.1.1.1 -Count 4 | Measure-Object -Property ResponseTime -Average).Average", "Average ping latency in ms (to 1.1.1.1)"),
            c("Get-NetAdapter | Where-Object Status -eq 'Up' | Format-Table Name, LinkSpeed -Auto", "Adapter link rate — this is the NIC's negotiated speed, NOT your internet speed"),
        ],
    },
    ShellTopic {
        topic: "users & accounts",
        keywords: &["user", "users", "account", "accounts", "whoami", "administrator", "group", "members", "permissions", "privilege"],
        note: None,
        commands: &[
            c("whoami /all", "Current user, groups, and privileges"),
            c("Get-LocalUser | Format-Table Name, Enabled, LastLogon -Auto", "Local user accounts"),
            c("Get-LocalGroupMember -Group 'Administrators'", "Who is a local Administrator"),
            c("Get-LocalGroup | Format-Table Name, Description -Auto", "Local groups"),
        ],
    },
    ShellTopic {
        topic: "disk & storage",
        keywords: &["disk", "drive", "storage", "free space", "volume", "partition", "ssd", "filesystem", "space"],
        note: None,
        commands: &[
            c(r"Get-Volume | Format-Table DriveLetter, FileSystemLabel, FileSystem, @{N='FreeGB';E={[int]($_.SizeRemaining/1GB)}}, @{N='SizeGB';E={[int]($_.Size/1GB)}} -Auto", "Volumes with free / total GB"),
            c(r"Get-PhysicalDisk | Format-Table FriendlyName, MediaType, @{N='SizeGB';E={[int]($_.Size/1GB)}}, HealthStatus -Auto", "Physical disks + health"),
            c(r"Get-PSDrive -PSProvider FileSystem | Format-Table Name, @{N='UsedGB';E={[int]($_.Used/1GB)}}, @{N='FreeGB';E={[int]($_.Free/1GB)}} -Auto", "Drive usage"),
        ],
    },
    ShellTopic {
        topic: "files, folders & search",
        keywords: &["file", "files", "folder", "directory", "search", "find", "grep", "content", "largest", "size", "recent files", "modified"],
        note: None,
        commands: &[
            c(r"Get-ChildItem -Path <dir> -Recurse -File | Sort-Object Length -Descending | Select-Object -First 15 FullName, @{N='MB';E={[int]($_.Length/1MB)}}", "Largest files under a directory"),
            c("Get-ChildItem -Path <dir> -Recurse -File | Where-Object LastWriteTime -gt (Get-Date).AddDays(-7) | Sort-Object LastWriteTime -Descending", "Files changed in the last week"),
            c(r"Select-String -Path '<dir>\*.txt' -Pattern '<regex>'", "Search file contents for a pattern"),
            c("Get-Content -Path <file> -Tail 50", "Last 50 lines of a file"),
        ],
    },
    ShellTopic {
        topic: "scheduled tasks & startup",
        keywords: &["scheduled task", "schtasks", "cron", "startup", "autorun", "task scheduler", "boot programs"],
        note: None,
        commands: &[
            c("Get-ScheduledTask | Where-Object State -ne 'Disabled' | Format-Table TaskName, State, TaskPath -Auto", "Enabled scheduled tasks"),
            c("Get-CimInstance Win32_StartupCommand | Format-Table Name, Command, Location -Auto", "Programs that run at startup"),
        ],
    },
    ShellTopic {
        topic: "installed software",
        keywords: &["installed", "software", "programs", "apps", "applications", "uninstall list", "winget", "appx"],
        note: None,
        commands: &[
            c(r"Get-ItemProperty HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*, HKLM:\Software\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall\* | Where-Object DisplayName | Select-Object DisplayName, DisplayVersion, Publisher | Sort-Object DisplayName", "Installed desktop programs"),
            c("winget list", "Packages known to winget (if installed)"),
            c("Get-AppxPackage | Select-Object Name, Version | Sort-Object Name", "Installed Store/UWP apps"),
        ],
    },
    ShellTopic {
        topic: "hardware (cpu / ram / gpu / bios)",
        keywords: &["hardware", "cpu", "processor", "ram", "memory", "gpu", "graphics", "bios", "motherboard", "device", "drivers"],
        note: None,
        commands: &[
            c("Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed", "CPU details"),
            c(r"Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum | ForEach-Object {[int]($_.Sum/1GB)}", "Total RAM (GB)"),
            c(r"Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, @{N='VRAM_GB';E={[int]($_.AdapterRAM/1GB)}}", "GPU(s) + driver"),
            c("Get-CimInstance Win32_BIOS | Select-Object Manufacturer, SMBIOSBIOSVersion, ReleaseDate", "BIOS info"),
        ],
    },
    ShellTopic {
        topic: "performance & resource usage",
        keywords: &["performance", "cpu load", "memory usage", "ram usage", "counter", "monitor", "utilization", "load"],
        note: None,
        commands: &[
            c(r"Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 3", "Live CPU utilization samples"),
            c(r"Get-CimInstance Win32_OperatingSystem | Select-Object @{N='FreeRAM_GB';E={[int]($_.FreePhysicalMemory/1MB)}}, @{N='TotalRAM_GB';E={[int]($_.TotalVisibleMemorySize/1MB)}}", "Free vs total RAM"),
            c(r"Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10 Name, @{N='MB';E={[int]($_.WorkingSet64/1MB)}}", "Top memory consumers"),
        ],
    },
    ShellTopic {
        topic: "firewall & security",
        keywords: &["firewall", "security", "defender", "antivirus", "blocked ports", "advfirewall", "rules"],
        note: None,
        commands: &[
            c("Get-NetFirewallProfile | Format-Table Name, Enabled -Auto", "Firewall profiles on/off"),
            c("Get-NetFirewallRule -Enabled True -Direction Inbound | Select-Object DisplayName, Action -First 30", "Enabled inbound firewall rules"),
            c("Get-MpComputerStatus | Select-Object AMServiceEnabled, RealTimeProtectionEnabled, AntivirusSignatureLastUpdated", "Windows Defender status"),
        ],
    },
    ShellTopic {
        topic: "environment variables",
        keywords: &["environment", "env", "path", "variable", "envvar", "%path%"],
        note: None,
        commands: &[
            c("Get-ChildItem Env: | Sort-Object Name | Format-Table -Auto", "All environment variables"),
            c("$env:PATH -split ';'", "PATH entries, one per line"),
        ],
    },
    ShellTopic {
        topic: "power & session control",
        keywords: &["shutdown", "restart", "reboot", "sleep", "lock", "logoff", "power"],
        note: Some("Shutdown/restart are hard-blocked by run_shell for safety; tell the user the command to run themselves if they want it."),
        commands: &[
            c("shutdown /s /t 0", "Shut down now (BLOCKED — user must run it)"),
            c("shutdown /r /t 0", "Restart now (BLOCKED — user must run it)"),
            c("rundll32.exe user32.dll,LockWorkStation", "Lock the screen"),
        ],
    },
];

/// Score a topic against a free-text query (keyword/topic/desc match).
fn score_topic(q: &str, t: &ShellTopic) -> u32 {
    let ql = q.to_lowercase();
    let terms: Vec<&str> =
        ql.split(|ch: char| !(ch.is_ascii_lowercase() || ch.is_ascii_digit())).filter(|w| w.chars().count() > 1).collect();
    let mut score = 0;
    for kw in t.keywords {
        if ql.contains(kw) || kw.contains(ql.as_str()) {
            score += 5;
        }
    }
    let topic = t.topic.to_lowercase();
    for term in &terms {
        if topic.contains(term) {
            score += 3;
        }
        if t.keywords.iter().any(|k| k.contains(term)) {
            score += 2;
        }
        if t.commands.iter().any(|c| c.desc.to_lowercase().contains(term) || c.cmd.to_lowercase().contains(term)) {
            score += 1;
        }
    }
    score
}

fn render_topic(t: &ShellTopic) -> String {
    let mut lines = vec![format!("## {}", t.topic)];
    if let Some(note) = t.note {
        lines.push(format!("note: {note}"));
    }
    for c in t.commands {
        lines.push(format!("  • {}\n    {}", c.desc, c.cmd));
    }
    // Small models tend to "rewrite" these into broken syntax (dropping the $ from
    // PowerShell variables, Python-style `x = ...`). Tell them to copy exactly.
    lines.push(
        "\nRun ONE of these with run_shell EXACTLY as written — copy it character-for-\
         character, keep every $ on PowerShell variables, do not rename, reformat, or \
         convert to another syntax."
            .to_string(),
    );
    lines.join("\n")
}

fn topic_index() -> String {
    SHELL_ENCYCLOPEDIA.iter().map(|t| format!("- {}", t.topic)).collect::<Vec<_>>().join("\n")
}

/// Look up the encyclopedia: the best-matching topic (plus a second if it's
/// nearly as relevant), rendered for the model; `None` when nothing matches.
pub fn lookup_shell_help(query: &str) -> Option<String> {
    let mut scored: Vec<(&ShellTopic, u32)> =
        SHELL_ENCYCLOPEDIA.iter().map(|t| (t, score_topic(query, t))).filter(|(_, s)| *s > 0).collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    let top = scored.first()?.1 as f64;
    let ranked: Vec<&ShellTopic> = scored
        .iter()
        .enumerate()
        .filter(|(i, (_, s))| *i == 0 || *s as f64 >= top * 0.8)
        .take(2)
        .map(|(_, (t, _))| *t)
        .collect();
    Some(ranked.iter().map(|t| render_topic(t)).collect::<Vec<_>>().join("\n\n"))
}

pub struct ShellHelpTool;

#[async_trait]
impl Tool for ShellHelpTool {
    fn name(&self) -> &str {
        "shell_help"
    }
    fn description(&self) -> &str {
        "Look up the EXACT, verified Windows PowerShell command(s) for a task before running anything \
         you're unsure about (login/event logs, processes, services, network, users, disk, files, \
         hardware, performance, firewall, scheduled tasks, software). Call with a topic/query; then run \
         the returned command with run_shell."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let query = if js_truthy(args.get("query")) {
            js_string(args.get("query"))
        } else if js_truthy(args.get("topic")) {
            js_string(args.get("topic"))
        } else {
            String::new()
        };
        let query = query.trim();
        if query.is_empty() {
            ctx.log(json!({ "tool": "shell_help", "query": "(index)" }));
            return ToolResult::ok(format!("Shell topics (query one for exact commands):\n{}", topic_index()));
        }
        // Return the best-matching topic only (plus a 2nd if it's nearly as relevant),
        // so the model gets a tight, actionable answer instead of a wall of text.
        match lookup_shell_help(query) {
            Some(out) => {
                ctx.log(json!({ "tool": "shell_help", "query": query, "matched": out.matches("## ").count() }));
                ToolResult::ok(out)
            }
            None => {
                ctx.log(json!({ "tool": "shell_help", "query": query, "matched": 0 }));
                ToolResult::ok(format!("No exact match for \"{query}\". Available topics:\n{}", topic_index()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_login_topic() {
        let out = lookup_shell_help("login logs last week").unwrap();
        assert!(out.starts_with("## login & logon history"));
        assert!(out.contains("Id=4624"));
        assert!(out.contains("EXACTLY as written"));
    }

    #[test]
    fn finds_wifi_and_memory_topics() {
        assert!(lookup_shell_help("wifi").unwrap().contains("netsh wlan show interfaces"));
        assert!(lookup_shell_help("top memory processes").unwrap().contains("WorkingSet64"));
    }

    #[test]
    fn speed_command_keeps_its_quotes() {
        let out = lookup_shell_help("internet speed").unwrap();
        assert!(out.contains(r#""{0:N1} Mbps down" -f"#));
    }

    #[tokio::test]
    async fn index_and_no_match() {
        let ctx = ToolContext::new(".");
        let r = ShellHelpTool.run(json!({}), &ctx).await;
        assert!(r.output.starts_with("Shell topics"));
        let r = ShellHelpTool.run(json!({ "query": "zzqx" }), &ctx).await;
        assert!(r.output.starts_with("No exact match for \"zzqx\""));
    }
}
