"""
Unique Celebration Visualizers for Each Ai|oS Security Tool
============================================================

Each tool gets its own COMPLETELY DIFFERENT dark/cyberpunk celebration aesthetic.
No generic confetti - each tool has unique particles matching its personality.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

# Unique celebration visualizers for each tool
TOOL_VISUALIZERS = {
    'dirreaper': {
        'name': 'DirReaper',
        'theme': 'Grim Reaper / Death',
        'particles': [
            'doll_head',      # Creepy doll heads with running mascara
            'dead_rose',      # Wilted black roses with falling petals
            'tombstone',      # Mini tombstones with RIP
            'skull',          # Floating skulls
            'ghost',          # Transparent spirits
            'scythe'          # Reaper scythes slashing
        ],
        'colors': ['#8800cc', '#aa33ff', '#660099', '#330066'],
        'sounds': ['scream', 'graveyard_bells', 'wind_howl']
    },

    'proxyphantom': {
        'name': 'ProxyPhantom',
        'theme': 'Phantom Fox / Spirit Fire',
        'particles': [
            'phantom_fox',    # Ghostly fox silhouettes
            'spirit_flame',   # Orange ethereal flames
            'fox_paw',        # Fox paw prints appearing
            'whisper',        # Ghostly whisper trails
            'orange_spark',   # Orange embers
            'shadow_orb'      # Dark orange orbs
        ],
        'colors': ['#ff6600', '#ffaa00', '#ff9933', '#cc5500'],
        'sounds': ['fox_howl', 'spirit_whisper', 'flame_crackle']
    },

    'vulnhunter': {
        'name': 'VulnHunter',
        'theme': 'Blood / Broken Defenses',
        'particles': [
            'blood_drop',     # Crimson blood drops
            'broken_shield',  # Shattering shields
            'cracked_screen', # Screen crack patterns
            'crosshair_hit',  # Targeting crosshairs hitting
            'warning_sign',   # Flashing warning triangles
            'shattered_lock'  # Broken padlocks
        ],
        'colors': ['#cc0000', '#ff3333', '#990000', '#660000'],
        'sounds': ['glass_shatter', 'alarm_blare', 'lock_break']
    },

    'payloadforge': {
        'name': 'PayloadForge',
        'theme': 'Lightning / Digital Corruption',
        'particles': [
            'lightning_bolt', # Electric lightning strikes
            'spark_shower',   # Electric sparks
            'glitch_block',   # Glitchy digital blocks
            'binary_rain',    # Corrupted binary code
            'power_surge',    # Energy surge waves
            'circuit_fry'     # Fried circuit patterns
        ],
        'colors': ['#ff00ff', '#cc00cc', '#ff33ff', '#990099'],
        'sounds': ['thunder_crack', 'electric_zap', 'power_surge']
    },

    'nmappro': {
        'name': 'NmapPro',
        'theme': 'Matrix Code / Network Nodes',
        'particles': [
            'matrix_char',    # Green falling characters
            'network_node',   # Network node connections
            'radar_ping',     # Radar pulse circles
            'data_packet',    # Network packet boxes
            'ip_address',     # Floating IP addresses
            'port_icon'       # Open port indicators
        ],
        'colors': ['#00ff88', '#00cc6f', '#00ff00', '#00aa44'],
        'sounds': ['radar_ping', 'data_flow', 'connection_established']
    },

    'aurorascan': {
        'name': 'AuroraScan',
        'theme': 'Northern Lights / Ice',
        'particles': [
            'aurora_wave',    # Aurora borealis waves
            'ice_crystal',    # Crystalline snowflakes
            'light_ribbon',   # Ribbon-like light streams
            'frost_particle', # Frost spreading patterns
            'star_glimmer',   # Twinkling stars
            'polar_light'     # Polar light beams
        ],
        'colors': ['#00ff88', '#00ffff', '#88ffaa', '#00ccaa'],
        'sounds': ['wind_chime', 'ice_crack', 'celestial_hum']
    },

    'spectratrace': {
        'name': 'SpectraTrace',
        'theme': 'Packet Fragments / Waveforms',
        'particles': [
            'packet_burst',   # Exploding packet fragments
            'waveform',       # Audio/signal waveforms
            'frequency_bar',  # Frequency spectrum bars
            'data_stream',    # Streaming data bytes
            'signal_pulse',   # Signal pulse rings
            'hex_fragment'    # Hexadecimal byte fragments
        ],
        'colors': ['#00aaff', '#0088cc', '#00ccff', '#0066aa'],
        'sounds': ['static_burst', 'frequency_sweep', 'data_capture']
    },

    'cipherspear': {
        'name': 'CipherSpear',
        'theme': 'Database Destruction / SQL Injection',
        'particles': [
            'db_table_shatter', # Database tables breaking
            'sql_symbol',       # SQL symbols (SELECT, WHERE, etc.)
            'injection_needle', # Syringe/needle injecting
            'data_leak',        # Data spilling out
            'corrupted_row',    # Corrupted database rows
            'spear_stab'        # Spear stabbing through
        ],
        'colors': ['#ff3333', '#cc0000', '#ff6666', '#aa0000'],
        'sounds': ['glass_shatter', 'data_corruption', 'injection_hiss']
    },

    'skybreaker': {
        'name': 'SkyBreaker',
        'theme': 'Wireless Waves / Signal Breaking',
        'particles': [
            'wifi_signal',    # WiFi signal bars breaking
            'radio_wave',     # Radio wave ripples
            'antenna_spark',  # Antenna sparking
            'frequency_crack',# Frequency lines cracking
            'wireless_burst', # Wireless explosion bursts
            'signal_jam'      # Signal jamming static
        ],
        'colors': ['#00ccff', '#0099cc', '#00ddff', '#0077aa'],
        'sounds': ['static_burst', 'frequency_modulation', 'signal_break']
    },

    'mythickey': {
        'name': 'MythicKey',
        'theme': 'Ancient Keys / Treasure',
        'particles': [
            'golden_key',     # Ornate golden keys
            'lock_mechanism', # Lock tumblers falling
            'treasure_coin',  # Ancient gold coins
            'chest_burst',    # Treasure chest opening
            'rune_symbol',    # Mystical runes
            'keyhole_glow'    # Glowing keyholes
        ],
        'colors': ['#ffd700', '#ffaa00', '#ffcc00', '#cc8800'],
        'sounds': ['lock_click', 'chest_open', 'coin_shower']
    },

    'nemesishydra': {
        'name': 'NemesisHydra',
        'theme': 'Multi-headed Beast / Venom',
        'particles': [
            'hydra_head',     # Serpent heads striking
            'snake_scale',    # Falling serpent scales
            'venom_drop',     # Dripping venom
            'fang_snap',      # Snapping fangs
            'tail_whip',      # Hydra tail strikes
            'poison_cloud'    # Toxic green clouds
        ],
        'colors': ['#ff0000', '#cc0000', '#00ff00', '#008800'],
        'sounds': ['hiss', 'venom_spit', 'scale_rattle']
    },

    'obsidianhunt': {
        'name': 'ObsidianHunt',
        'theme': 'Stone / Fortress Walls',
        'particles': [
            'obsidian_shard', # Black obsidian shards
            'stone_crack',    # Cracking stone patterns
            'fortress_wall',  # Wall segments falling
            'rock_dust',      # Stone dust clouds
            'shield_emblem',  # Defensive shield icons
            'armor_piece'     # Armor fragments
        ],
        'colors': ['#444444', '#666666', '#222222', '#888888'],
        'sounds': ['stone_grind', 'fortress_crumble', 'metal_clang']
    },

    'vectorflux': {
        'name': 'VectorFlux',
        'theme': 'Dimensional Portals / Flux',
        'particles': [
            'portal_ring',    # Swirling portal rings
            'vector_arrow',   # Directional vector arrows
            'flux_particle',  # Flux energy particles
            'dimension_tear', # Reality tears/rifts
            'warp_trail',     # Warp speed trails
            'quantum_bubble'  # Quantum probability bubbles
        ],
        'colors': ['#9933ff', '#7700cc', '#aa44ff', '#6600aa'],
        'sounds': ['portal_open', 'dimension_shift', 'flux_warble']
    }
}

# CSS/JS code generator for each visualizer type
def generate_visualizer_css_js(tool_key):
    """Generate unique CSS/JS for a specific tool's visualizer."""

    if tool_key not in TOOL_VISUALIZERS:
        return ""

    config = TOOL_VISUALIZERS[tool_key]

    # Base template with tool-specific implementations
    css_js = f"""
/* ============================================
   {config['name']} - Unique Celebration Visualizer
   Theme: {config['theme']}
   ============================================ */

<style>
.celebration-overlay-{tool_key} {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 9999;
    overflow: hidden;
}}
"""

    # Generate particle-specific CSS based on tool type
    if tool_key == 'dirreaper':
        css_js += """
/* Doll Head with Running Mascara */
.particle-doll_head {
    position: absolute;
    font-size: 40px;
    animation: doll-fall 4s ease-in forwards;
    filter: grayscale(80%) contrast(120%);
}

.particle-doll_head::after {
    content: '';
    position: absolute;
    width: 2px;
    height: 50px;
    background: linear-gradient(to bottom, #000, transparent);
    left: 50%;
    top: 100%;
    animation: mascara-run 4s ease-out;
}

@keyframes doll-fall {
    0% { transform: translate(0, 0) rotate(0deg) scale(1); opacity: 1; }
    100% { transform: translate(var(--tx), var(--ty)) rotate(720deg) scale(0); opacity: 0; }
}

@keyframes mascara-run {
    0% { height: 0; }
    100% { height: 30px; }
}

/* Dead Rose */
.particle-dead_rose {
    position: absolute;
    width: 30px;
    height: 40px;
    background: radial-gradient(circle, #1a0000, #000);
    border-radius: 50% 50% 45% 45%;
    animation: rose-wilt 5s ease-in-out forwards;
    box-shadow: 0 0 10px rgba(139, 0, 0, 0.5);
}

.particle-dead_rose::before {
    content: '🥀';
    position: absolute;
    font-size: 30px;
    filter: grayscale(100%) brightness(50%);
}

@keyframes rose-wilt {
    0% { transform: translate(0, 0) rotate(0deg) scale(1); opacity: 1; }
    100% { transform: translate(var(--tx), var(--ty)) rotate(360deg) scale(0); opacity: 0; }
}

/* Tombstone */
.particle-tombstone {
    position: absolute;
    width: 40px;
    height: 60px;
    background: linear-gradient(135deg, #333, #111);
    border-radius: 50% 50% 0 0;
    animation: tombstone-drop 3s ease-in forwards;
    border: 2px solid #666;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.particle-tombstone::after {
    content: '⚰️';
    font-size: 20px;
}

@keyframes tombstone-drop {
    0% { transform: translateY(-50px) rotate(0deg); opacity: 1; }
    80% { transform: translateY(85vh) rotate(360deg); }
    90% { transform: translateY(95vh) rotate(360deg); }
    100% { transform: translateY(100vh) rotate(360deg); opacity: 0; }
}

/* Skull */
.particle-skull {
    position: absolute;
    font-size: 35px;
    animation: skull-float 4s ease-in-out forwards;
    filter: drop-shadow(0 0 10px #8800cc);
}

@keyframes skull-float {
    0% { transform: translateY(-30px) rotate(-10deg); opacity: 1; }
    50% { transform: translateY(50vh) rotate(10deg); }
    100% { transform: translateY(100vh) rotate(-10deg); opacity: 0; }
}

/* Ghost */
.particle-ghost {
    position: absolute;
    font-size: 40px;
    animation: ghost-haunt 5s ease-in-out forwards;
    opacity: 0.7;
}

@keyframes ghost-haunt {
    0% { transform: translate(-50px, -50px); opacity: 0; }
    25% { transform: translate(20px, 30vh); opacity: 0.7; }
    50% { transform: translate(-20px, 60vh); opacity: 0.5; }
    75% { transform: translate(30px, 85vh); opacity: 0.3; }
    100% { transform: translate(-10px, 100vh); opacity: 0; }
}

/* Scythe */
.particle-scythe {
    position: absolute;
    font-size: 50px;
    animation: scythe-slash 2s ease-out forwards;
    filter: drop-shadow(0 0 15px #aa33ff);
}

@keyframes scythe-slash {
    0% { transform: translate(-100px, -50px) rotate(0deg); opacity: 1; }
    50% { transform: translate(50vw, 50vh) rotate(360deg); opacity: 1; }
    100% { transform: translate(100vw, 100vh) rotate(720deg); opacity: 0; }
}
"""

    elif tool_key == 'proxyphantom':
        css_js += """
/* Phantom Fox */
.particle-phantom_fox {
    position: absolute;
    font-size: 45px;
    animation: fox-dash 3s ease-out forwards;
    filter: drop-shadow(0 0 20px #ff6600);
    opacity: 0.8;
}

@keyframes fox-dash {
    0% { transform: translateX(-100px) scaleX(-1); opacity: 0; }
    30% { transform: translateX(30vw) scaleX(-1); opacity: 0.8; }
    70% { transform: translateX(70vw) scaleX(1); opacity: 0.6; }
    100% { transform: translateX(120vw) scaleX(1); opacity: 0; }
}

/* Spirit Flame */
.particle-spirit_flame {
    position: absolute;
    width: 20px;
    height: 40px;
    background: linear-gradient(to top, #ff6600, #ffaa00, transparent);
    border-radius: 50% 50% 20% 20%;
    animation: flame-flicker 2s ease-in-out forwards;
    box-shadow: 0 0 20px #ff6600, 0 0 40px #ffaa00;
}

@keyframes flame-flicker {
    0%, 100% { transform: translateY(-20px) scale(1); opacity: 1; }
    25% { transform: translateY(25vh) scale(1.2); }
    50% { transform: translateY(50vh) scale(0.8); }
    75% { transform: translateY(75vh) scale(1.1); }
    100% { transform: translateY(100vh) scale(0); opacity: 0; }
}

/* Fox Paw */
.particle-fox_paw {
    position: absolute;
    font-size: 30px;
    animation: paw-step 3s linear forwards;
    filter: drop-shadow(0 0 10px #ff9933);
}

@keyframes paw-step {
    0% { transform: translate(0, -30px) rotate(0deg); opacity: 0; }
    20% { opacity: 1; }
    80% { opacity: 0.5; }
    100% { transform: translate(0, 100vh) rotate(720deg); opacity: 0; }
}

/* Whisper Trail */
.particle-whisper {
    position: absolute;
    font-size: 25px;
    color: #ffaa00;
    animation: whisper-fade 4s ease-out forwards;
    opacity: 0.6;
}

@keyframes whisper-fade {
    0% { transform: translate(-50px, -30px); opacity: 0; }
    30% { transform: translate(20px, 40vh); opacity: 0.6; }
    70% { transform: translate(-30px, 80vh); opacity: 0.3; }
    100% { transform: translate(50px, 100vh); opacity: 0; }
}

/* Orange Spark */
.particle-orange_spark {
    position: absolute;
    width: 8px;
    height: 8px;
    background: radial-gradient(circle, #ffaa00, #ff6600);
    border-radius: 50%;
    animation: spark-burst 2s ease-out forwards;
    box-shadow: 0 0 10px #ff6600;
}

@keyframes spark-burst {
    0% { transform: translate(0, 0) scale(1); opacity: 1; }
    100% { transform: translate(var(--tx), var(--ty)) scale(0); opacity: 0; }
}

/* Shadow Orb */
.particle-shadow_orb {
    position: absolute;
    width: 40px;
    height: 40px;
    background: radial-gradient(circle, #cc5500, #000);
    border-radius: 50%;
    animation: orb-float 4s ease-in-out forwards;
    box-shadow: 0 0 30px #ff6600;
}

@keyframes orb-float {
    0% { transform: translate(-50px, -50px) scale(0); opacity: 0; }
    30% { transform: translate(30vw, 30vh) scale(1); opacity: 1; }
    70% { transform: translate(60vw, 70vh) scale(0.8); opacity: 0.7; }
    100% { transform: translate(100vw, 100vh) scale(0); opacity: 0; }
}
"""

    elif tool_key == 'vulnhunter':
        css_js += """
/* Blood Drop */
.particle-blood_drop {
    position: absolute;
    width: 12px;
    height: 16px;
    background: radial-gradient(ellipse at 50% 40%, #ff3333, #990000);
    border-radius: 50% 50% 50% 0;
    transform: rotate(45deg);
    animation: blood-drip 3s ease-in forwards;
    box-shadow: 0 0 5px #cc0000;
}

@keyframes blood-drip {
    0% { transform: translateY(-30px) rotate(45deg); opacity: 1; }
    100% { transform: translateY(100vh) rotate(45deg); opacity: 0; }
}

/* Broken Shield */
.particle-broken_shield {
    position: absolute;
    font-size: 40px;
    animation: shield-shatter 2s ease-out forwards;
    filter: drop-shadow(0 0 10px #ff3333);
}

@keyframes shield-shatter {
    0% { transform: scale(1) rotate(0deg); opacity: 1; }
    50% { transform: scale(1.5) rotate(180deg); opacity: 0.8; }
    100% { transform: scale(0) rotate(360deg); opacity: 0; }
}

/* Cracked Screen */
.particle-cracked_screen {
    position: absolute;
    width: 100px;
    height: 100px;
    background: linear-gradient(45deg, transparent 48%, #ff3333 48%, #ff3333 52%, transparent 52%),
                linear-gradient(-45deg, transparent 48%, #ff3333 48%, #ff3333 52%, transparent 52%);
    animation: crack-spread 1.5s ease-out forwards;
    opacity: 0.7;
}

@keyframes crack-spread {
    0% { transform: scale(0); opacity: 0; }
    50% { transform: scale(1.5); opacity: 0.7; }
    100% { transform: scale(3); opacity: 0; }
}

/* Crosshair Hit */
.particle-crosshair_hit {
    position: absolute;
    width: 50px;
    height: 50px;
    border: 3px solid #cc0000;
    border-radius: 50%;
    animation: crosshair-pulse 1s ease-out forwards;
}

.particle-crosshair_hit::before,
.particle-crosshair_hit::after {
    content: '';
    position: absolute;
    background: #ff3333;
}

.particle-crosshair_hit::before {
    width: 100%;
    height: 2px;
    top: 50%;
    left: 0;
}

.particle-crosshair_hit::after {
    width: 2px;
    height: 100%;
    left: 50%;
    top: 0;
}

@keyframes crosshair-pulse {
    0% { transform: scale(0); opacity: 1; box-shadow: 0 0 0 0 #ff3333; }
    50% { transform: scale(1); opacity: 1; box-shadow: 0 0 20px 10px #ff3333; }
    100% { transform: scale(2); opacity: 0; box-shadow: 0 0 40px 20px #ff3333; }
}

/* Warning Sign */
.particle-warning_sign {
    position: absolute;
    font-size: 50px;
    animation: warning-flash 2s ease-in-out forwards;
    filter: drop-shadow(0 0 15px #cc0000);
}

@keyframes warning-flash {
    0%, 100% { opacity: 0; transform: scale(0) rotate(0deg); }
    25% { opacity: 1; transform: scale(1.2) rotate(10deg); }
    50% { opacity: 1; transform: scale(1) rotate(-10deg); }
    75% { opacity: 1; transform: scale(1.2) rotate(10deg); }
}

/* Shattered Lock */
.particle-shattered_lock {
    position: absolute;
    font-size: 40px;
    animation: lock-break 1.5s ease-out forwards;
    filter: drop-shadow(0 0 10px #990000);
}

@keyframes lock-break {
    0% { transform: scale(1) rotate(0deg); opacity: 1; }
    30% { transform: scale(1.1) rotate(-5deg); }
    60% { transform: scale(0.9) rotate(5deg); }
    100% { transform: scale(0) rotate(180deg); opacity: 0; }
}
"""

    # Close CSS and add JavaScript for particle spawning
    css_js += """
</style>

<script>
"""

    # Generate JS spawner for this specific tool
    css_js += f"""
const {config['name']}Visualizer = {{
    overlay: null,
    particles: {config['particles']},
    colors: {config['colors']},

    init() {{
        if (!this.overlay) {{
            this.overlay = document.createElement('div');
            this.overlay.className = 'celebration-overlay-{tool_key}';
            document.body.appendChild(this.overlay);
        }}
    }},

    spawn(particleType, count = 30) {{
        this.init();

        for (let i = 0; i < count; i++) {{
            const particle = document.createElement('div');
            particle.className = `particle-${{particleType}}`;
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 0.5 + 's';

            // Tool-specific emoji/content
            if (particleType === 'doll_head') {{
                particle.innerHTML = '🎎';
                particle.style.fontSize = (30 + Math.random() * 20) + 'px';
            }} else if (particleType === 'dead_rose') {{
                // Rose styling applied via CSS
            }} else if (particleType === 'skull') {{
                particle.innerHTML = '💀';
            }} else if (particleType === 'ghost') {{
                particle.innerHTML = '👻';
            }} else if (particleType === 'scythe') {{
                particle.innerHTML = '⚰️';
            }} else if (particleType === 'phantom_fox') {{
                particle.innerHTML = '🦊';
            }} else if (particleType === 'fox_paw') {{
                particle.innerHTML = '🐾';
            }} else if (particleType === 'whisper') {{
                particle.innerHTML = '💨';
            }}

            this.overlay.appendChild(particle);
            setTimeout(() => particle.remove(), 5000);
        }}
    }},

    celebrateBig() {{
        // Spawn all particle types for HUGE celebration
        this.particles.forEach(type => {{
            this.spawn(type, 15);
        }});
    }},

    celebrateMedium() {{
        // Spawn 2-3 particle types
        const types = this.particles.slice(0, 3);
        types.forEach(type => this.spawn(type, 10));
    }},

    celebrateSmall() {{
        // Spawn 1 particle type
        this.spawn(this.particles[0], 5);
    }}
}};

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {{
    {config['name']}Visualizer.init();
    console.log('[{config['name']}] Unique visualizer loaded! 🎮');
}});
</script>
"""

    return css_js

# Export functions
def get_visualizer_code(tool_key):
    """Get the complete CSS/JS visualizer code for a tool."""
    return generate_visualizer_css_js(tool_key)

def list_all_visualizers():
    """List all available visualizers."""
    return {k: v['theme'] for k, v in TOOL_VISUALIZERS.items()}
