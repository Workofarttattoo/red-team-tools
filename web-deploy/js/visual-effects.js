/**
 * Red Team Tools - Visual Effects Library
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
 *
 * Cyberpunk visual effects for pentest tool pages
 */

// Matrix Rain Effect
class MatrixRain {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.columns = [];
        this.fontSize = options.fontSize || 14;
        this.color = options.color || '#00ff00';
        this.opacity = options.opacity || 0.15;
        this.speed = options.speed || 50;

        this.chars = 'ｦｱｳｴｵｶｷｹｺｻｼｽｾｿﾀﾂﾃﾅﾆﾇﾈﾊﾋﾎﾏﾐﾑﾒﾓﾔﾕﾗﾘﾜ0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<>{}[]()!@#$%^&*';

        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.columnCount = Math.floor(this.canvas.width / this.fontSize);

        if (this.columns.length === 0) {
            for (let i = 0; i < this.columnCount; i++) {
                this.columns[i] = Math.random() * this.canvas.height / this.fontSize;
            }
        }
    }

    animate() {
        this.ctx.fillStyle = `rgba(0, 0, 0, ${this.opacity})`;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.fillStyle = this.color;
        this.ctx.font = `${this.fontSize}px monospace`;

        for (let i = 0; i < this.columns.length; i++) {
            const char = this.chars[Math.floor(Math.random() * this.chars.length)];
            const x = i * this.fontSize;
            const y = this.columns[i] * this.fontSize;

            this.ctx.fillText(char, x, y);

            if (y > this.canvas.height && Math.random() > 0.95) {
                this.columns[i] = 0;
            }

            this.columns[i]++;
        }

        setTimeout(() => requestAnimationFrame(() => this.animate()), this.speed);
    }
}

// Scanning Grid Effect
class ScanningGrid {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.scanLines = [];
        this.radarAngle = 0;

        // Create scan lines
        for (let i = 0; i < 3; i++) {
            this.scanLines.push({
                y: Math.random() * this.canvas.height,
                speed: 2 + Math.random() * 3,
                color: ['#ff0055', '#00ff00', '#00d4ff'][i],
                opacity: 0.6
            });
        }

        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw grid
        this.ctx.strokeStyle = 'rgba(0, 255, 0, 0.03)';
        this.ctx.lineWidth = 1;

        for (let x = 0; x < this.canvas.width; x += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }

        for (let y = 0; y < this.canvas.height; y += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }

        // Draw scan lines
        this.scanLines.forEach(line => {
            const gradient = this.ctx.createLinearGradient(0, line.y - 20, 0, line.y + 20);
            gradient.addColorStop(0, 'transparent');
            gradient.addColorStop(0.5, line.color);
            gradient.addColorStop(1, 'transparent');

            this.ctx.strokeStyle = gradient;
            this.ctx.lineWidth = 2;
            this.ctx.globalAlpha = line.opacity;

            this.ctx.beginPath();
            this.ctx.moveTo(0, line.y);
            this.ctx.lineTo(this.canvas.width, line.y);
            this.ctx.stroke();

            line.y += line.speed;
            if (line.y > this.canvas.height) {
                line.y = -20;
            }
        });

        this.ctx.globalAlpha = 1;

        // Draw radar sweep
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const radius = Math.min(this.canvas.width, this.canvas.height) * 0.6;

        const gradient = this.ctx.createConicGradient(this.radarAngle, centerX, centerY);
        gradient.addColorStop(0, 'rgba(0, 255, 0, 0)');
        gradient.addColorStop(0.1, 'rgba(0, 255, 0, 0.2)');
        gradient.addColorStop(0.2, 'rgba(0, 255, 0, 0)');

        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.radarAngle += 0.02;

        requestAnimationFrame(() => this.animate());
    }
}

// Network Packet Effect
class NetworkPackets {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.packets = [];
        this.maxPackets = 15;
        this.protocols = ['TCP', 'UDP', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'FTP'];
        this.colors = {
            'TCP': '#00ff00',
            'UDP': '#00d4ff',
            'HTTP': '#ffaa00',
            'HTTPS': '#ff0055',
            'DNS': '#a855f7',
            'SSH': '#00ff88',
            'FTP': '#ff3333'
        };

        this.spawnInterval = setInterval(() => this.spawnPacket(), 1000);
    }

    spawnPacket() {
        if (this.packets.length >= this.maxPackets) {
            const oldPacket = this.packets.shift();
            if (oldPacket.element.parentNode) {
                oldPacket.element.remove();
            }
        }

        const protocol = this.protocols[Math.floor(Math.random() * this.protocols.length)];
        const size = Math.floor(Math.random() * 1500) + 64;
        const fromTop = Math.random() > 0.5;

        const packet = document.createElement('div');
        packet.className = 'network-packet';
        packet.style.cssText = `
            position: fixed;
            ${fromTop ? 'top' : 'bottom'}: ${Math.random() * 80 + 10}%;
            ${fromTop ? 'right' : 'left'}: -100px;
            background: ${this.colors[protocol]};
            color: #000;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.8rem;
            font-weight: bold;
            box-shadow: 0 0 20px ${this.colors[protocol]};
            z-index: 3;
            pointer-events: none;
            white-space: nowrap;
        `;
        packet.textContent = `${protocol} ${size}B`;

        this.container.appendChild(packet);

        const duration = 3000 + Math.random() * 2000;
        const distance = window.innerWidth + 200;

        packet.animate([
            { transform: `translateX(0)`, opacity: 0 },
            { transform: `translateX(${fromTop ? '-' : ''}${distance}px)`, opacity: 1, offset: 0.1 },
            { transform: `translateX(${fromTop ? '-' : ''}${distance * 2}px)`, opacity: 0 }
        ], {
            duration: duration,
            easing: 'linear'
        });

        setTimeout(() => {
            if (packet.parentNode) {
                packet.remove();
            }
            const index = this.packets.indexOf(packet);
            if (index > -1) {
                this.packets.splice(index, 1);
            }
        }, duration);

        this.packets.push({ element: packet });
    }

    destroy() {
        clearInterval(this.spawnInterval);
        this.packets.forEach(p => {
            if (p.element.parentNode) {
                p.element.remove();
            }
        });
    }
}

// Live Terminal Effect
class LiveTerminal {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.createTerminal();
        this.commands = [];
        this.currentLine = 0;
        this.minimized = false;
    }

    createTerminal() {
        this.terminalEl = document.createElement('div');
        this.terminalEl.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 500px;
            background: rgba(0, 0, 0, 0.95);
            border: 1px solid #00ff00;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0, 255, 0, 0.3);
            z-index: 100;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            transition: all 0.3s;
        `;

        this.terminalEl.innerHTML = `
            <div style="background: rgba(0, 255, 0, 0.2); padding: 8px 12px; border-bottom: 1px solid #00ff00; display: flex; justify-content: space-between; align-items: center; cursor: pointer;" onclick="window.liveTerminal.toggle()">
                <span style="color: #00ff00; font-weight: bold;">root@pentest:~#</span>
                <span style="color: #00ff00; cursor: pointer;">━</span>
            </div>
            <div style="padding: 12px; max-height: 300px; overflow-y: auto; color: #0f0;" id="liveTerminalContent"></div>
        `;

        this.container.appendChild(this.terminalEl);
        this.content = document.getElementById('liveTerminalContent');

        window.liveTerminal = this;
    }

    toggle() {
        this.minimized = !this.minimized;
        if (this.minimized) {
            this.terminalEl.style.height = '40px';
            this.content.style.display = 'none';
        } else {
            this.terminalEl.style.height = 'auto';
            this.content.style.display = 'block';
        }
    }

    addLine(text, color = '#0f0') {
        const line = document.createElement('div');
        line.style.cssText = `margin-bottom: 4px; color: ${color}; animation: terminalFadeIn 0.2s;`;
        line.textContent = text;
        this.content.appendChild(line);

        while (this.content.children.length > 30) {
            this.content.removeChild(this.content.firstChild);
        }

        this.content.scrollTop = this.content.scrollHeight;
    }

    startDemoCommands(tool) {
        const commands = this.getCommandsForTool(tool);
        let index = 0;

        const runNext = () => {
            if (index < commands.length) {
                const cmd = commands[index];
                this.addLine(`# ${cmd.command}`, '#00d4ff');

                setTimeout(() => {
                    cmd.output.forEach(line => {
                        this.addLine(line, cmd.color || '#0f0');
                    });

                    index++;
                    setTimeout(runNext, 3000 + Math.random() * 2000);
                }, 500);
            } else {
                setTimeout(() => {
                    this.content.innerHTML = '';
                    index = 0;
                    setTimeout(runNext, 2000);
                }, 5000);
            }
        };

        setTimeout(runNext, 2000);
    }

    getCommandsForTool(tool) {
        const commandSets = {
            sqlmap: [
                {
                    command: 'sqlmap -u "https://target.com/page?id=1" --dbs',
                    output: [
                        '[*] Testing connection to target',
                        '[+] Parameter \'id\' is injectable',
                        '[+] DBMS: MySQL 5.7.x',
                        '[+] Databases: webapp_db, users_db, admin'
                    ]
                },
                {
                    command: 'sqlmap -u "https://target.com/page?id=1" -D webapp_db --tables',
                    output: [
                        '[*] Fetching tables from webapp_db',
                        '[+] Tables: users, sessions, products',
                        '[+] Found 3 tables'
                    ]
                }
            ],
            nmap: [
                {
                    command: 'nmap -sS -p- 192.168.1.100',
                    output: [
                        'Starting Nmap scan...',
                        'Discovered open port 22/tcp on 192.168.1.100',
                        'Discovered open port 80/tcp on 192.168.1.100',
                        'Discovered open port 443/tcp on 192.168.1.100',
                        '[+] Scan complete: 3 ports open'
                    ]
                }
            ],
            default: [
                {
                    command: 'whoami',
                    output: ['root']
                },
                {
                    command: 'uname -a',
                    output: ['Linux pentest 5.15.0 x86_64 GNU/Linux']
                }
            ]
        };

        return commandSets[tool] || commandSets.default;
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes terminalFadeIn {
        from { opacity: 0; transform: translateX(-5px); }
        to { opacity: 1; transform: translateX(0); }
    }
`;
document.head.appendChild(style);

// Export for use
window.VisualEffects = {
    MatrixRain,
    ScanningGrid,
    NetworkPackets,
    LiveTerminal
};
