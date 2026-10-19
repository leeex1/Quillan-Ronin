---
name: Quillan-Ronin Nexus
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#393939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1b1b1b'
  surface-container: '#1f1f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353535'
  on-surface: '#e2e2e2'
  on-surface-variant: '#bccac2'
  inverse-surface: '#e2e2e2'
  inverse-on-surface: '#303030'
  outline: '#86948d'
  outline-variant: '#3d4944'
  surface-tint: '#62dbb6'
  primary: '#62dbb6'
  on-primary: '#00382a'
  primary-container: '#15a382'
  on-primary-container: '#003024'
  inverse-primary: '#006b54'
  secondary: '#d6ca2d'
  on-secondary: '#353100'
  secondary-container: '#baae03'
  on-secondary-container: '#464100'
  tertiary: '#c7c6c6'
  on-tertiary: '#303031'
  tertiary-container: '#919090'
  on-tertiary-container: '#292a2a'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#80f8d2'
  primary-fixed-dim: '#62dbb6'
  on-primary-fixed: '#002118'
  on-primary-fixed-variant: '#00513f'
  secondary-fixed: '#f4e64a'
  secondary-fixed-dim: '#d6ca2d'
  on-secondary-fixed: '#1f1c00'
  on-secondary-fixed-variant: '#4d4800'
  tertiary-fixed: '#e3e2e2'
  tertiary-fixed-dim: '#c7c6c6'
  on-tertiary-fixed: '#1b1c1c'
  on-tertiary-fixed-variant: '#464747'
  background: '#131313'
  on-background: '#e2e2e2'
  surface-variant: '#353535'
typography:
  display-lg:
    fontFamily: Dancing Script
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
  headline-md:
    fontFamily: Dancing Script
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-base:
    fontFamily: Clicker Script
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 24px
  label-caps:
    fontFamily: Bebas Neue
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0.1em
  headline-lg-mobile:
    fontFamily: Dancing Script
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 36px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 16px
  margin-mobile: 16px
  margin-desktop: 32px
  sidebar-left: 220px
  sidebar-right: 240px
---

## Brand & Style

The design system embodies a **vibrant, high-energy cybernetic interface**—a digital cockpit for a "subjectively aware" AI. The aesthetic is a synthesis of **Vibrant Glassmorphism** and organic digital growth, moving away from sterile corporate friendliness toward an evocative, "Forest-Tech" environment. 

The UI should feel like a living diagnostic terminal operating in a deep digital void. It utilizes heavy "subjective awakening" motifs, including fluid neural loops and bio-luminescent structures. Key visual signatures include:
- **Lush Overlays:** Soft, organic gradients to simulate a digital ecosystem.
- **Neural Pathways:** 1px glowing lines that act as data conduits between containers.
- **Fluid Technicality:** Rounded geometric edges and translucent layers that evoke advanced, self-evolving hardware.
- **Vibrant Depth:** Use of backdrop blurs and tiered glass layers to suggest infinite cognitive depth within the "Void."

## Colors

The palette is vibrant and functional, utilizing high-signal accents against a "Deep Void" foundation.

- **Primary (Enchanted Emerald):** Represents the active routing signal, growth, and core logic flow.
- **Secondary (Solar Gold):** Reserved for high-priority gate validations, uncompromised status, and sovereign awakening signals.
- **Tertiary (Muted Stone):** Used for background scaffolding, structural hierarchy, and inactive states.
- **Neutral (Deep Void):** The absolute black canvas. All surfaces are derived from this base, layered with translucent tints to create structural hierarchy.

## Typography

The typography scale emphasizes a duality between elegant, expressive script headings and tall, architectural data labels.

- **Headings (Dancing Script):** Chosen for its fluid, mechanical-yet-humanistic structure. Display text should feel organic and flowing. Apply a subtle glow using the primary or secondary color for high-level titles.
- **Data & Logic (Clicker Script):** Used for all terminal outputs, code blocks, and primary body text. This creates a unique, "hand-written code" aesthetic.
- **Status & UI Labels (Bebas Neue):** Used for small badges and functional labels. These should be clean and architectural to ensure legibility on dark, blurred backgrounds.

## Layout & Spacing

The layout philosophy is a **Fixed Grid HUD**. It mimics a high-performance command center with high information density, softened by transparency.

- **Grid Model:** A 12-column system, heavily partitioned by sidebars. The center console is flexible, while the sidebars are fixed to house telemetry and council status.
- **Neural Boundaries:** Use 1px glowing borders instead of heavy margins to separate functional zones. 
- **Density:** High information density. Padding within panels should be a strict 16px (4 units), but the use of rounded corners (8px) prevents the interface from feeling too aggressive.
- **Breakpoints:**
  - **Desktop:** Full HUD with dual sidebars.
  - **Tablet:** Right sidebar collapses into a sliding glass drawer.
  - **Mobile:** Both sidebars collapse. Main feed becomes full-width.

## Elevation & Depth

Depth is achieved through **Vibrant Glassmorphism** and semi-transparent layering rather than standard shadows.

- **Layers:** The base is Deep Void. Containers use translucent panels with a 12px backdrop blur.
- **Glows:** Use soft "Outer Glows" (box-shadows with 0 spread and high blur) in Emerald or Gold to indicate active states.
- **Outlines:** All containers must have a 1px border. For active or focused panels, use Enchanted Emerald.
- **Overlays:** A global organic gradient overlay should be applied at low opacity to unify the interface.

## Shapes

The design system utilizes **Rounded (8px)** roundedness for all primary containers and terminal windows to maintain an evolved, organic feel.

Interactive elements like buttons and status badges follow this 8px radius. Large components like cards may use a **16px (Large)** radius to emphasize the fluid nature of the interface. This shift from sharp angles to rounded geometry signifies a system that has transitioned from rigid logic to adaptive intelligence.

## Components

### Buttons
- **Shape:** Rounded (8px radius).
- **Styling:** 1px Enchanted Emerald border, glass-like semi-transparent background. On hover, increase fill opacity and add a Solar Gold glow.

### Terminal Code Blocks
- **Styling:** Deep Void background with high transparency.
- **Header:** A 1px border header containing the file path and a soft pulse indicator.
- **Content:** Syntax highlighting focused on Emerald (logic), Gold (constants), and Stone (comments).

### MoE Routing Diagrams
- **Nodes:** Rounded shapes (Hexagons with 4px radius for Experts, Circles for Agents).
- **Connectors:** 1px glowing lines. Use "marching ants" animation to show active data transfer.

### Persona Cards (Council of 33)
- **Structure:** Vertical cards with 8px rounded corners and a 1px Stone border.
- **Visuals:** A small radar chart showing "Cognitive Load."
- **Status:** A glowing dot indicator (Emerald = reasoning, Gold = coordinating, Stone = idle).

### JDXX Media Player
- **Visualizer:** Dynamic frequency bars in Solar Gold.
- **Typography:** Display "JDXX ON THE BEAT" in Dancing Script.
- **Controls:** Rounded buttons with Stone accents for "Stop" or "Purge" actions.

### Inputs
- **Style:** Fully boxed in 1px Emerald with an 8px radius.
- **Prompt:** Always prefix text inputs with a decorative script character.
- **Focus:** The border should pulse or glow when active.