---
name: Obsidian Prism
colors:
  surface: '#0e1511'
  surface-dim: '#0e1511'
  surface-bright: '#343b36'
  surface-container-lowest: '#09100c'
  surface-container-low: '#161d19'
  surface-container: '#1a211d'
  surface-container-high: '#242c27'
  surface-container-highest: '#2f3632'
  on-surface: '#dde4dd'
  on-surface-variant: '#bbcabf'
  inverse-surface: '#dde4dd'
  inverse-on-surface: '#2b322d'
  outline: '#86948a'
  outline-variant: '#3c4a42'
  surface-tint: '#4edea3'
  primary: '#4edea3'
  on-primary: '#003824'
  primary-container: '#10b981'
  on-primary-container: '#00422b'
  inverse-primary: '#006c49'
  secondary: '#4cd7f6'
  on-secondary: '#003640'
  secondary-container: '#03b5d3'
  on-secondary-container: '#00424e'
  tertiary: '#ffb3af'
  on-tertiary: '#650911'
  tertiary-container: '#fc7c78'
  on-tertiary-container: '#711419'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#6ffbbe'
  primary-fixed-dim: '#4edea3'
  on-primary-fixed: '#002113'
  on-primary-fixed-variant: '#005236'
  secondary-fixed: '#acedff'
  secondary-fixed-dim: '#4cd7f6'
  on-secondary-fixed: '#001f26'
  on-secondary-fixed-variant: '#004e5c'
  tertiary-fixed: '#ffdad7'
  tertiary-fixed-dim: '#ffb3af'
  on-tertiary-fixed: '#410005'
  on-tertiary-fixed-variant: '#842225'
  background: '#0e1511'
  on-background: '#dde4dd'
  surface-variant: '#2f3632'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.05em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 40px
  sidebar-width: 280px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 32px
---

## Brand & Style
This design system is engineered for high-performance monitoring and complex data visualization. The brand personality is technical, elite, and cinematic, evoking the feel of a premium command center. It targets professional users who require clarity in high-density information environments.

The visual style is a refined **Glassmorphism** set against a **Minimalist** dark canvas. It utilizes "Obsidian" surfaces that feel like polished volcanic glass, layered with varying levels of transparency and background blurs to create a sense of depth without clutter. High-frequency data is punctuated by vibrant, glowing accents that serve as functional indicators of system health and status.

## Colors
The palette is anchored in **Deep Obsidian (#070a13)**, providing a pure, non-distracting foundation. 

*   **Healthy Maize Green (#10b981):** The primary action and "success" color. It should glow slightly when used in status indicators.
*   **Disease Cyan/Blue (#06b6d4):** Used for informational data, secondary navigation, and calm metrics.
*   **Indicator Orange (#f97316):** Reserved for warnings, active "damage" states, or pending actions.
*   **Larva Red (#ef4444):** High-priority alerts, critical errors, and destructive actions.

Surfaces use a semi-transparent white or primary-tinted overlay (Alpha 4% to 8%) to create the glass effect over the obsidian background.

## Typography
The system uses **Inter** for all primary communication to ensure maximum readability and a clean, modernist aesthetic. For technical data, telemetry, and status labels, **JetBrains Mono** is employed to provide a "developer-tool" feel that emphasizes precision.

All typography should maintain high contrast against the dark background. Use `white` for primary headers and `neutral (#94a3b8)` for secondary body text. Labels and monospaced text should often adopt the color of their status (e.g., a green label for a healthy metric).

## Layout & Spacing
The layout follows a **Dashboard-centric** model with a fixed left-hand sidebar for primary navigation. 

*   **Grid:** A 12-column fluid grid for the main content area.
*   **Sidebar:** Fixed 280px width, utilizing a frosted glass effect with a `1px` right-side border.
*   **Responsiveness:** 
    *   **Desktop (1200px+):** Full 12-column display with 32px margins.
    *   **Tablet (768px - 1199px):** Sidebar collapses into an icon-only rail (80px) or hides behind a burger menu. Content margins reduce to 24px.
    *   **Mobile (<768px):** Single column stack. Navigation moves to a bottom bar or top-level menu. Typography scales down (e.g., `headline-lg` becomes `headline-lg-mobile`).

## Elevation & Depth
Depth is achieved through **Backdrop Blurs** and **Tonal Layering** rather than traditional shadows.

1.  **Level 0 (Background):** Solid `#070a13`.
2.  **Level 1 (Cards/Sidebar):** Background: `rgba(255, 255, 255, 0.04)`, Backdrop-filter: `blur(12px)`, Border: `1px solid rgba(255, 255, 255, 0.1)`.
3.  **Level 2 (Popovers/Modals):** Background: `rgba(255, 255, 255, 0.08)`, Backdrop-filter: `blur(20px)`, Border: `1px solid rgba(255, 255, 255, 0.2)`. 

**Glowing Accents:** Elements with critical status use a `drop-shadow` with the status color (e.g., `0 0 15px rgba(16, 185, 129, 0.3)`) to simulate light emission from the glass surface.

## Shapes
The design system uses a **Rounded (0.5rem)** base. This balances the technical, sharp nature of the data with a modern, premium feel. 

*   **Buttons & Inputs:** 0.5rem (8px).
*   **Cards & Containers:** 1rem (16px).
*   **Status Tags/Chips:** Pill-shaped (Full round) to distinguish them from interactive buttons.

## Components

*   **Buttons:** Primary buttons use a solid maize green fill with black text for maximum contrast. Secondary buttons are "ghost" style with a fine `1px` border and subtle hover glow.
*   **Translucent Cards:** The core container. Must have a subtle inner glow (top-left light source) and a 1px stroke. The background blur is essential to maintain legibility over background patterns.
*   **Inputs:** Dark, recessed backgrounds (`rgba(0,0,0,0.2)`) with a `1px` maize green bottom border when focused.
*   **Chips/Status Indicators:** Small, pill-shaped badges. Use a low-opacity fill of the status color with high-opacity text (e.g., Dark Red background with Bright Red text).
*   **Sidebar Links:** Active states use a maize green "indicator" vertical bar on the left and a subtle gradient fade from the left.
*   **Data Visualizations:** Charts should use the indicator colors (Cyan, Green, Orange, Red). Grid lines in charts should be kept to a minimum, using `rgba(255, 255, 255, 0.05)`.