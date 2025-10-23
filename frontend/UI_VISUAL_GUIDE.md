# 🎨 RealyticsAI UI Visual Guide

## Complete Interface Layout

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         STICKY HEADER (#121212)                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  rAI                             Property Intelligence Suite       │  ║
║  │  └─┬┘                                                     (small)  │  ║
║  │    └─ #00A99D (teal)                                               │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                           MAIN CHAT AREA                                 ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  FEATURE BUTTONS (Horizontal Row)                                  │  ║
║  │  ┌──────┐ ┌──────────────────┐ ┌──────────────┐ ┌──────────────┐ │  ║
║  │  │ ✨ │ │ 🏠 Property    │ │ 💰 Price     │ │ 🤝 AI        │ │  ║
║  │  │Auto │ │    Discovery   │ │   Valuation  │ │  Negotiation │ │  ║
║  │  └──────┘ └──────────────────┘ └──────────────┘ └──────────────┘ │  ║
║  │  (Active: #00A99D border + bg, Inactive: #333 border)             │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  USER MESSAGE BUBBLE (#1e1e1e, border #333)                        │  ║
║  │  "Find me 3 BHK apartments in Whitefield"                          │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  AI MESSAGE BUBBLE (#161616, border #333)                          │  ║
║  │                                                                     │  ║
║  │  Great news! I found **351 properties** for you. Here are the...   │  ║
║  │                                                                     │  ║
║  │  ┌────────────────────────────────────────────────────────────────┐│  ║
║  │  │  PROPERTY CARD 1 (#1A1A1A, border #333, rounded)              ││  ║
║  │  │  ┌──────────────────────────────────────────────────────────┐ ││  ║
║  │  │  │                                                          │ ││  ║
║  │  │  │         HIGH-RESOLUTION IMAGE (18rem height)            │ ││  ║
║  │  │  │         Modern house from Unsplash                      │ ││  ║
║  │  │  │                                                          │ ││  ║
║  │  │  └──────────────────────────────────────────────────────────┘ ││  ║
║  │  │  ┌──────────────────────────────────────────────────────────┐ ││  ║
║  │  │  │  3 BHK Apartment in Whitefield Old    ← Bold White      │ ││  ║
║  │  │  │  💰 Price: ₹31.9 Lakhs               ← Normal           │ ││  ║
║  │  │  │  📐 Size: 1796 sqft | 🚿 3 Bath | 🪟 1 Balcony          │ ││  ║
║  │  │  │                                       ↑ Gray #EAEAEA     │ ││  ║
║  │  │  └──────────────────────────────────────────────────────────┘ ││  ║
║  │  └────────────────────────────────────────────────────────────────┘│  ║
║  │                                                                     │  ║
║  │  ┌────────────────────────────────────────────────────────────────┐│  ║
║  │  │  PROPERTY CARD 2 (same layout)                                ││  ║
║  │  └────────────────────────────────────────────────────────────────┘│  ║
║  │                                                                     │  ║
║  │  ┌────────────────────────────────────────────────────────────────┐│  ║
║  │  │  PROPERTY CARD 3 (same layout)                                ││  ║
║  │  └────────────────────────────────────────────────────────────────┘│  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                         STICKY INPUT AREA (bottom)                       ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │  ┌─────────────────────────────────────────────────┐  ┌─────────┐ │  ║
║  │  │  Ask about properties, prices, or negotiations  │  │  Send   │ │  ║
║  │  │  (#1e1e1e bg, #333 border)                      │  │ #00A99D │ │  ║
║  │  └─────────────────────────────────────────────────┘  └─────────┘ │  ║
║  │                Press Enter to send (hint text)                     │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Color Palette

```css
/* Primary Colors */
--bg-primary:    #121212  /* Main background */
--bg-secondary:  #161616  /* AI message bubbles */
--bg-tertiary:   #1A1A1A  /* Property cards */
--bg-input:      #1e1e1e  /* Input field, user messages */

/* Accent & Borders */
--accent:        #00A99D  /* Brand teal - buttons, logo "AI" */
--border:        #333333  /* Subtle borders everywhere */

/* Text Colors */
--text-primary:   #FFFFFF  /* Headings, bold text */
--text-secondary: #EAEAEA  /* Property details */
--text-tertiary:  #CCCCCC  /* Regular text */
--text-muted:     #888888  /* Hints, labels */
```

---

## Property Card Breakdown

```
╔═══════════════════════════════════════════════╗
║  PROPERTY CARD (#1A1A1A, border #333)        ║
║  ┌──────────────────────────────────────────┐ ║
║  │                                          │ ║  ← 18rem height
║  │                                          │ ║    (288px)
║  │         IMAGE TOP HALF                   │ ║    Covers 60%
║  │    (backgroundImage, cover, center)      │ ║    of card
║  │                                          │ ║
║  └──────────────────────────────────────────┘ ║
║  ┌──────────────────────────────────────────┐ ║
║  │  DESCRIPTION BOTTOM (padding: 1rem)     │ ║  ← Auto height
║  │                                          │ ║    Minimum 6rem
║  │  ╔════════════════════════════════════╗ │ ║
║  │  ║ Title (1.125rem, bold, #fff)      ║ │ ║
║  │  ║ "3 BHK Apartment in Whitefield"   ║ │ ║
║  │  ╚════════════════════════════════════╝ │ ║
║  │                                          │ ║
║  │  💰 Price: ₹31.9 Lakhs                  │ ║  ← 0.9rem, #ccc
║  │                                          │ ║
║  │  📐 Size: 1796 sqft | 🚿 3 Bath |      │ ║  ← 0.85rem,
║  │  🪟 1 Balcony                           │ ║    #EAEAEA
║  └──────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════╝
```

---

## Intent Detection Visual

```
USER TYPES                        SYSTEM DETECTS         UI SHOWS
───────────────────────────────────────────────────────────────────────
"Find apartments"          →      RECOMMENDATION    →    Property Cards
"Show me 3 BHK"            →      RECOMMENDATION    →    Property Cards
"Under 50 lakhs"           →      RECOMMENDATION    →    Property Cards

"What is the price?"       →      PREDICTION        →    Text with **Bold**
"Estimate value"           →      PREDICTION        →    Price breakdown
"How much does it cost?"   →      PREDICTION        →    Formatted text

"Help me negotiate"        →      NEGOTIATION       →    Strategy text
"Best offer?"              →      NEGOTIATION       →    Recommendations
"Counter strategy"         →      NEGOTIATION       →    Guidance

"Hello"                    →      AUTO              →    General response
"What can you do?"         →      AUTO              →    Feature list
```

---

## Response Format Examples

### 1. Recommendation Response (with cards)
```javascript
{
  intent: 'RECOMMENDATION',
  text: 'Great news! I found **351 properties** for you.',
  properties: [
    {
      title: '3 BHK Apartment in Whitefield Old',
      price_lakhs: 31.9,
      total_sqft: 1796,
      bath: 3,
      balcony: 1,
      image: 'https://...'
    }
    // ... 2-3 more
  ]
}
```

### 2. Prediction Response (text only)
```javascript
{
  intent: 'PREDICTION',
  text: `**Price Valuation Ready**

I've analyzed the property parameters and generated a **data-driven estimate**.

**Estimated Price:** ₹59.35 Lakhs
**Likely Range:** ₹53.4 - ₹65.3 Lakhs

This valuation is based on **location, BHK, size (sqft), and amenities**.`
}
```

### 3. Negotiation Response (text only)
```javascript
{
  intent: 'NEGOTIATION',
  text: `**AI Negotiation Activated**

I can draft a persuasive offer, set a **target price**, and recommend **counter strategies**.

**Suggested Starting Offer:** ₹48 Lakhs
**Maximum Budget:** ₹55 Lakhs
**Strategy:** Start 15% below asking price, emphasize immediate purchase readiness.`
}
```

---

## Typography Scale

```
Logo "rAI"               → 1.5rem (24px) bold
Feature Buttons          → 0.875rem (14px)
Message Text             → 0.95rem (15.2px) line-height 1.6
Property Card Title      → 1.125rem (18px) bold
Property Card Price      → 0.9rem (14.4px)
Property Card Details    → 0.85rem (13.6px)
Input Placeholder        → 0.95rem (15.2px)
Hint Text               → 0.75rem (12px)
```

---

## Spacing System

```
Padding Levels:
- Cards:          1rem (16px)
- Messages:       1.25rem (20px)
- Input:          0.75rem (12px)
- Feature Buttons: 0.5rem 1rem (8px 16px)

Gap Levels:
- Feature Buttons: 0.5rem (8px)
- Property Cards:  1rem (16px)
- Messages:        1.5rem (24px)

Border Radius:
- Cards:           0.75rem (12px)
- Buttons:         9999px (pill shape)
- Input:           0.75rem (12px)
- Message Bubbles: 1rem (16px)
```

---

## Interactive States

### Feature Buttons
```css
/* Default */
border: 1px solid #333
color: #ccc
background: transparent

/* Hover */
border: 1px solid #00A99D
color: #00A99D

/* Active */
border: 1px solid #00A99D
color: #00A99D
background: rgba(0, 169, 157, 0.1)
```

### Send Button
```css
/* Disabled (no text) */
background: #333
color: #fff
cursor: not-allowed

/* Enabled (has text) */
background: #00A99D
color: #fff
cursor: pointer

/* Hover */
background: #00BFB0 (slightly lighter)
```

---

## Mobile Responsiveness

The component is fully responsive:
- Max width: **48rem (768px)** for optimal readability
- Mobile: Buttons stack vertically if needed
- Cards: Full width on mobile
- Input: Flexbox ensures proper layout

---

## Accessibility Features

✅ Keyboard Navigation (Enter to send)
✅ Disabled state for empty input
✅ Clear visual feedback on hover
✅ High contrast text (#fff on #121212)
✅ Semantic HTML structure
✅ Focus states on interactive elements

---

**Visual Style**: ChatGPT / Gemini-inspired  
**Dark Mode**: Primary (#121212)  
**Accent Color**: Teal (#00A99D)  
**Layout**: Image-first property cards  
**Status**: ✅ Production Ready
