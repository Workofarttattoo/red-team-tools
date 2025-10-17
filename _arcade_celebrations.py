"""
Arcade Celebration System for Ai|oS Security Toolkit
====================================================

Provides arcade-style visual celebrations for major security milestones:
- Flashing lights
- Showering coins
- Metallic confetti
- Fireworks
- Initials entry (like classic arcade high scores)

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

# JavaScript/CSS for arcade celebrations - to be embedded in tool GUIs
ARCADE_CELEBRATION_JS = """
<style>
/* Celebration overlay */
.celebration-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 9999;
    overflow: hidden;
}

/* Confetti particles */
.confetti {
    position: absolute;
    width: 10px;
    height: 10px;
    background: #ff00ff;
    animation: confetti-fall 3s linear forwards;
}

@keyframes confetti-fall {
    to {
        transform: translateY(100vh) rotate(360deg);
        opacity: 0;
    }
}

/* Coin shower */
.coin {
    position: absolute;
    width: 30px;
    height: 30px;
    background: radial-gradient(circle, #ffd700, #ffaa00);
    border-radius: 50%;
    border: 2px solid #ff8800;
    animation: coin-fall 2s ease-in forwards;
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.8);
}

@keyframes coin-fall {
    0% {
        transform: translateY(-50px) rotateY(0deg);
        opacity: 1;
    }
    100% {
        transform: translateY(100vh) rotateY(720deg);
        opacity: 0;
    }
}

/* Firework particles */
.firework {
    position: absolute;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    animation: firework-explode 1s ease-out forwards;
}

@keyframes firework-explode {
    0% {
        transform: translate(0, 0);
        opacity: 1;
    }
    100% {
        transform: translate(var(--tx), var(--ty));
        opacity: 0;
    }
}

/* Flashing screen border */
.screen-flash {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 9998;
    animation: flash-border 0.5s ease-in-out 3;
    border: 10px solid transparent;
}

@keyframes flash-border {
    0%, 100% { border-color: transparent; }
    50% { border-color: var(--flash-color, #00ff00); }
}

/* Big win modal */
.big-win-modal {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) scale(0);
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.95), rgba(20, 20, 20, 0.95));
    border: 5px solid var(--win-color, #ffd700);
    border-radius: 20px;
    padding: 40px;
    z-index: 10000;
    text-align: center;
    box-shadow: 0 0 100px var(--win-color, #ffd700);
    animation: big-win-appear 0.5s ease-out forwards;
}

@keyframes big-win-appear {
    to {
        transform: translate(-50%, -50%) scale(1);
    }
}

.big-win-title {
    font-size: 60px;
    font-weight: 900;
    color: var(--win-color, #ffd700);
    text-shadow: 0 0 30px var(--win-color, #ffd700);
    animation: pulse-glow 1s infinite alternate;
    margin-bottom: 20px;
}

@keyframes pulse-glow {
    from { text-shadow: 0 0 30px var(--win-color, #ffd700); }
    to { text-shadow: 0 0 60px var(--win-color, #ffd700), 0 0 90px var(--win-color, #ffd700); }
}

.big-win-message {
    font-size: 24px;
    color: #fff;
    margin-bottom: 30px;
}

.big-win-stats {
    font-size: 18px;
    color: #aaa;
    margin-bottom: 20px;
}

/* Initials entry (arcade style) */
.initials-entry {
    margin-top: 30px;
    padding: 20px;
    background: rgba(0, 0, 0, 0.5);
    border-radius: 10px;
    border: 2px solid var(--win-color, #ffd700);
}

.initials-prompt {
    font-size: 20px;
    color: var(--win-color, #ffd700);
    margin-bottom: 15px;
    font-family: 'Courier New', monospace;
    text-transform: uppercase;
}

.initials-input {
    display: flex;
    justify-content: center;
    gap: 10px;
}

.initial-char {
    width: 60px;
    height: 80px;
    background: rgba(0, 0, 0, 0.8);
    border: 3px solid var(--win-color, #ffd700);
    border-radius: 8px;
    font-size: 48px;
    font-weight: 900;
    color: var(--win-color, #ffd700);
    text-align: center;
    font-family: 'Courier New', monospace;
    text-transform: uppercase;
    box-shadow: inset 0 0 20px rgba(255, 215, 0, 0.3);
}

.initials-buttons {
    margin-top: 20px;
    display: flex;
    justify-content: center;
    gap: 15px;
}

.initials-btn {
    padding: 12px 30px;
    background: linear-gradient(135deg, var(--win-color, #ffd700), #ff8800);
    border: none;
    border-radius: 8px;
    color: #000;
    font-weight: 700;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s;
    text-transform: uppercase;
}

.initials-btn:hover {
    transform: scale(1.1);
    box-shadow: 0 0 30px var(--win-color, #ffd700);
}

/* Close button */
.close-celebration {
    margin-top: 20px;
    padding: 12px 40px;
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 8px;
    color: #fff;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.3s;
}

.close-celebration:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.5);
}

/* Verbose activity log styles */
.verbose-log {
    background: rgba(0, 0, 0, 0.8);
    border: 2px solid rgba(0, 255, 0, 0.3);
    border-radius: 10px;
    padding: 15px;
    max-height: 300px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    margin: 20px 0;
}

.verbose-log-line {
    margin: 3px 0;
    padding: 2px 5px;
    animation: fade-in 0.3s;
}

@keyframes fade-in {
    from { opacity: 0; transform: translateX(-10px); }
    to { opacity: 1; transform: translateX(0); }
}

.log-info { color: #00ff00; }
.log-success { color: #00ffff; font-weight: bold; }
.log-warning { color: #ffaa00; }
.log-error { color: #ff5555; font-weight: bold; }
.log-critical { color: #ff0000; font-weight: 900; animation: blink 0.5s infinite; }

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
</style>

<script>
// Arcade celebration system
const ArcadeCelebrations = {
    // Main celebration overlay
    overlay: null,

    // Initialize overlay
    init() {
        if (!this.overlay) {
            this.overlay = document.createElement('div');
            this.overlay.className = 'celebration-overlay';
            document.body.appendChild(this.overlay);
        }
    },

    // Spawn confetti
    spawnConfetti(count = 100) {
        this.init();
        const colors = ['#ff00ff', '#00ffff', '#ffff00', '#ff0000', '#00ff00', '#ffd700'];

        for (let i = 0; i < count; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.style.left = Math.random() * 100 + '%';
            confetti.style.background = colors[Math.floor(Math.random() * colors.length)];
            confetti.style.animationDelay = Math.random() * 0.5 + 's';
            confetti.style.animationDuration = (Math.random() * 2 + 2) + 's';
            this.overlay.appendChild(confetti);

            setTimeout(() => confetti.remove(), 5000);
        }
    },

    // Spawn coin shower
    spawnCoins(count = 50) {
        this.init();

        for (let i = 0; i < count; i++) {
            const coin = document.createElement('div');
            coin.className = 'coin';
            coin.innerHTML = '💰';
            coin.style.left = Math.random() * 100 + '%';
            coin.style.animationDelay = Math.random() * 0.3 + 's';
            this.overlay.appendChild(coin);

            setTimeout(() => coin.remove(), 3000);
        }

        // Play coin sound (if audio enabled)
        this.playSound('coin');
    },

    // Create firework explosion
    createFirework(x, y, color) {
        this.init();
        const particles = 30;

        for (let i = 0; i < particles; i++) {
            const particle = document.createElement('div');
            particle.className = 'firework';
            particle.style.left = x + 'px';
            particle.style.top = y + 'px';
            particle.style.background = color;

            const angle = (Math.PI * 2 * i) / particles;
            const distance = 100 + Math.random() * 100;
            const tx = Math.cos(angle) * distance;
            const ty = Math.sin(angle) * distance;

            particle.style.setProperty('--tx', tx + 'px');
            particle.style.setProperty('--ty', ty + 'px');

            this.overlay.appendChild(particle);
            setTimeout(() => particle.remove(), 1000);
        }
    },

    // Spawn random fireworks
    spawnFireworks(count = 5) {
        const colors = ['#ff00ff', '#00ffff', '#ffff00', '#ff0000', '#00ff00', '#ffd700'];

        for (let i = 0; i < count; i++) {
            setTimeout(() => {
                const x = Math.random() * window.innerWidth;
                const y = Math.random() * (window.innerHeight / 2);
                const color = colors[Math.floor(Math.random() * colors.length)];
                this.createFirework(x, y, color);
            }, i * 300);
        }

        this.playSound('firework');
    },

    // Screen flash
    screenFlash(color = '#00ff00', times = 3) {
        const flash = document.createElement('div');
        flash.className = 'screen-flash';
        flash.style.setProperty('--flash-color', color);
        flash.style.animationIterationCount = times;
        document.body.appendChild(flash);

        setTimeout(() => flash.remove(), times * 500);
    },

    // Big win modal
    showBigWin(title, message, stats, color = '#ffd700', enableInitials = false) {
        const modal = document.createElement('div');
        modal.className = 'big-win-modal';
        modal.style.setProperty('--win-color', color);

        let initialsHTML = '';
        if (enableInitials) {
            initialsHTML = `
                <div class="initials-entry">
                    <div class="initials-prompt">Enter Your Initials</div>
                    <div class="initials-input">
                        <input type="text" maxlength="1" class="initial-char" id="initial1" value="A">
                        <input type="text" maxlength="1" class="initial-char" id="initial2" value="A">
                        <input type="text" maxlength="1" class="initial-char" id="initial3" value="A">
                    </div>
                    <div class="initials-buttons">
                        <button class="initials-btn" onclick="ArcadeCelebrations.saveInitials()">💾 SAVE</button>
                    </div>
                </div>
            `;
        }

        modal.innerHTML = `
            <div class="big-win-title">${title}</div>
            <div class="big-win-message">${message}</div>
            <div class="big-win-stats">${stats}</div>
            ${initialsHTML}
            <button class="close-celebration" onclick="this.parentElement.remove()">CONTINUE</button>
        `;

        document.body.appendChild(modal);

        // Auto-focus first initial input
        if (enableInitials) {
            setTimeout(() => {
                const input1 = document.getElementById('initial1');
                if (input1) input1.focus();
            }, 500);
        }

        // Play big win sound
        this.playSound('bigwin');
    },

    // Save initials to localStorage
    saveInitials() {
        const i1 = document.getElementById('initial1')?.value || 'A';
        const i2 = document.getElementById('initial2')?.value || 'A';
        const i3 = document.getElementById('initial3')?.value || 'A';
        const initials = (i1 + i2 + i3).toUpperCase();

        const scores = JSON.parse(localStorage.getItem('aios_highscores') || '[]');
        scores.push({
            initials: initials,
            timestamp: new Date().toISOString(),
            tool: window.CURRENT_TOOL || 'Unknown'
        });
        localStorage.setItem('aios_highscores', JSON.stringify(scores));

        alert(`Initials ${initials} saved! You're now in the hall of fame! 🏆`);
    },

    // Play sound effect (if audio system available)
    playSound(type) {
        // Audio would be implemented with Web Audio API or audio tags
        console.log(`[AUDIO] Playing ${type} sound effect`);
    },

    // Full celebration combo (the works!)
    celebrateFullCombo(title, message, stats, color = '#ffd700') {
        this.screenFlash(color, 5);
        this.spawnConfetti(150);
        this.spawnCoins(75);
        this.spawnFireworks(8);

        setTimeout(() => {
            this.showBigWin(title, message, stats, color, true);
        }, 1000);
    },

    // Medium celebration
    celebrateMedium(message, color = '#00ff00') {
        this.screenFlash(color, 2);
        this.spawnConfetti(50);
        this.spawnFireworks(3);
        this.showBigWin('SUCCESS!', message, '', color, false);
    },

    // Minor celebration
    celebrateMinor(message) {
        this.spawnConfetti(20);
        console.log(`[CELEBRATION] ${message}`);
    }
};

// Verbose logging system
const VerboseLogger = {
    logContainer: null,

    init(containerId = 'verboseLog') {
        this.logContainer = document.getElementById(containerId);
        if (!this.logContainer) {
            this.logContainer = document.createElement('div');
            this.logContainer.id = containerId;
            this.logContainer.className = 'verbose-log';
            // Container should be added to page by tool
        }
    },

    log(message, type = 'info') {
        if (!this.logContainer) this.init();

        const line = document.createElement('div');
        line.className = `verbose-log-line log-${type}`;
        line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;

        this.logContainer.appendChild(line);
        this.logContainer.scrollTop = this.logContainer.scrollHeight;

        // Also log to console
        console.log(`[VERBOSE:${type.toUpperCase()}] ${message}`);
    },

    info(message) { this.log(message, 'info'); },
    success(message) { this.log(message, 'success'); },
    warning(message) { this.log(message, 'warning'); },
    error(message) { this.log(message, 'error'); },
    critical(message) { this.log(message, 'critical'); }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    ArcadeCelebrations.init();
    console.log('[ARCADE] Celebration system loaded! 🎮');
});
</script>
"""

# Python helper functions for triggering celebrations from backend
def get_celebration_js():
    """Return the JavaScript celebration code to embed in tool GUIs."""
    return ARCADE_CELEBRATION_JS

def celebration_trigger_code(event_type, **kwargs):
    """
    Generate JavaScript code to trigger specific celebration.

    Args:
        event_type: Type of celebration (shell_obtained, cve_critical, exploit_success, etc.)
        **kwargs: Additional parameters for celebration

    Returns:
        JavaScript code as string
    """
    celebrations = {
        'shell_obtained': {
            'title': '🎯 SHELL OBTAINED!',
            'message': 'Remote shell access achieved!',
            'color': '#00ff00',
            'level': 'full'
        },
        'root_access': {
            'title': '👑 ROOT ACCESS!',
            'message': 'Maximum privilege level achieved!',
            'color': '#ffd700',
            'level': 'full'
        },
        'cve_critical': {
            'title': '🚨 CRITICAL CVE FOUND!',
            'message': kwargs.get('cve', 'Critical vulnerability discovered!'),
            'color': '#ff0000',
            'level': 'full'
        },
        'cve_high': {
            'title': '⚠️ HIGH SEVERITY!',
            'message': kwargs.get('cve', 'High severity vulnerability found!'),
            'color': '#ff6600',
            'level': 'medium'
        },
        'exploit_success': {
            'title': '💥 EXPLOIT SUCCESS!',
            'message': 'Target successfully exploited!',
            'color': '#ff00ff',
            'level': 'medium'
        },
        'scan_complete': {
            'title': '✅ SCAN COMPLETE!',
            'message': f"Found {kwargs.get('count', 0)} issues",
            'color': '#00ffff',
            'level': 'minor'
        },
        'directory_found': {
            'title': '📁 DIRECTORY FOUND!',
            'message': kwargs.get('path', 'Hidden directory discovered!'),
            'color': '#aa33ff',
            'level': 'minor'
        }
    }

    event = celebrations.get(event_type, celebrations['scan_complete'])
    stats = kwargs.get('stats', '')

    if event['level'] == 'full':
        return f"ArcadeCelebrations.celebrateFullCombo('{event['title']}', '{event['message']}', '{stats}', '{event['color']}');"
    elif event['level'] == 'medium':
        return f"ArcadeCelebrations.celebrateMedium('{event['message']}', '{event['color']}');"
    else:
        return f"ArcadeCelebrations.celebrateMinor('{event['message']}');"

def verbose_log_code(message, level='info'):
    """Generate JavaScript for verbose logging."""
    return f"VerboseLogger.{level}('{message}');"
