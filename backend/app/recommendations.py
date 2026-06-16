"""
Site-specific pest management recommendations.

Guidance sourced from:
  - FAO Technical Guide on Fall Armyworm Management (2018)
  - CABI Invasive Species Compendium — Fall Armyworm
  - CIMMYT Integrated Pest Management Guidelines for Maize
  - TAAT (Technologies for African Agricultural Transformation) FAW briefs

Each detection class maps to a structured management protocol with cultural,
biological, and chemical control tiers plus long-term prevention advice.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Region-specific advisory data (keyed by broad region)
# ---------------------------------------------------------------------------

REGION_ADVISORY: Dict[str, Dict[str, str]] = {
    "sub_saharan_africa": {
        "region_name": "Sub-Saharan Africa",
        "advisory": (
            "Contact your local agricultural extension officer for nationally "
            "registered pesticides. In East Africa, community-based push-pull "
            "technology (ICIPE/Rothamsted) has shown 80%+ reduction in FAW damage."
        ),
        "emergency_contact": "FAO Regional Office for Africa: +233-302-610930",
    },
    "south_asia": {
        "region_name": "South Asia",
        "advisory": (
            "Consult your state agriculture department for approved insecticide "
            "list. Pheromone traps (5 per hectare) are recommended for monitoring "
            "adult moth flights."
        ),
        "emergency_contact": "FAO Regional Office for Asia: +66-2-697-4000",
    },
    "americas": {
        "region_name": "Americas",
        "advisory": (
            "Bt maize hybrids provide partial resistance. Scout fields weekly and "
            "apply threshold-based decisions. Contact your county extension agent."
        ),
        "emergency_contact": "FAO Regional Office for Latin America: +56-2-2923-2100",
    },
    "default": {
        "region_name": "General",
        "advisory": (
            "Consult the FAO Fall Armyworm Monitoring and Early Warning System "
            "(FAMEWS) app and your national plant protection organisation for "
            "locally registered control products."
        ),
        "emergency_contact": "FAO HQ Emergency: +39-06-57051",
    },
}


def _resolve_region(latitude: Optional[float], longitude: Optional[float]) -> str:
    """Map GPS coordinates to a broad advisory region.

    This is a simplified classifier. A production system would use a proper
    reverse-geocoding service or country-boundary lookup.
    """
    if latitude is None or longitude is None:
        return "default"

    # Very rough bounding boxes
    if -35 <= latitude <= 37 and -20 <= longitude <= 55:
        return "sub_saharan_africa"
    if 5 <= latitude <= 40 and 60 <= longitude <= 100:
        return "south_asia"
    if -55 <= latitude <= 50 and -130 <= longitude <= -30:
        return "americas"

    return "default"


# ---------------------------------------------------------------------------
# Per-class management protocols
# ---------------------------------------------------------------------------

MANAGEMENT_PROTOCOLS: Dict[str, dict] = {
    "fall-armyworm-egg": {
        "severity": "low",
        "alert_color": "amber",
        "display_name": "Fall Armyworm — Egg Mass",
        "description": (
            "Egg masses detected on leaf surfaces. Each mass can contain "
            "100–200 eggs covered in a fuzzy, felt-like layer. Hatching "
            "occurs in 2–3 days under warm conditions."
        ),
        "cultural_control": [
            "Scout fields every 3–5 days, focusing on leaf undersides of young plants.",
            "Physically remove and crush egg masses by hand where feasible.",
            "Destroy volunteer maize and weed hosts around field borders.",
        ],
        "biological_control": [
            "Encourage egg parasitoids such as Trichogramma spp. and Telenomus remus by maintaining hedgerows and wildflower strips.",
            "Release Telenomus remus where commercially available (10,000 per hectare).",
            "Avoid broad-spectrum insecticides that harm natural enemies.",
        ],
        "chemical_control": [
            "Chemical intervention is generally NOT recommended at the egg stage.",
            "If heavy egg laying is observed across >50% of plants, prepare for targeted larval-stage spraying.",
        ],
        "prevention": [
            "Plant early in the season to avoid peak moth flight periods.",
            "Use pheromone traps (5 per hectare) to monitor adult moth activity and predict egg-laying peaks.",
            "Intercrop maize with non-host crops (beans, groundnuts) to disrupt oviposition.",
        ],
        "sources": [
            "FAO, 2018 — Integrated Management of Fall Armyworm on Maize: A Guide for FFS in Africa",
            "CABI, 2023 — Fall Armyworm (Spodoptera frugiperda) Datasheet",
        ],
    },
    "fall-armyworm-frass": {
        "severity": "medium",
        "alert_color": "amber",
        "display_name": "Fall Armyworm — Frass Detected",
        "description": (
            "Sawdust-like frass (larval excrement) visible in the leaf whorl. "
            "This confirms active feeding by larvae hidden inside the whorl. "
            "Immediate inspection is needed."
        ),
        "cultural_control": [
            "Peel back leaves and inspect whorls for hidden larvae.",
            "Apply a mixture of sand or fine soil with wood ash directly into the whorl (suffocates early-instar larvae).",
            "Remove and destroy heavily infested plants to reduce field-level populations.",
        ],
        "biological_control": [
            "Apply Bacillus thuringiensis var. kurstaki (Bt) directly into whorls — most effective against 1st–3rd instar larvae.",
            "Spray neem-based biopesticides (azadirachtin 0.03%) early morning when larvae are feeding.",
            "Promote natural enemies: encourage ants (Pheidole spp.) and earwigs that prey on larvae in whorls.",
        ],
        "chemical_control": [
            "If >20% of plants show fresh frass, targeted chemical control may be justified.",
            "Apply emamectin benzoate (0.5 g a.i./L) or chlorantraniliprole as a whorl application.",
            "Spray late afternoon or early evening when larvae move to feed.",
            "Always rotate modes of action (IRAC groups) to prevent resistance.",
        ],
        "prevention": [
            "Implement push-pull intercropping: Desmodium (silverleaf) between rows repels moths, Brachiaria border grass attracts and traps them.",
            "Rotate maize with non-host crops each season.",
            "Destroy crop residues immediately after harvest to break the life cycle.",
        ],
        "sources": [
            "FAO, 2018 — Integrated Management of Fall Armyworm on Maize",
            "CIMMYT, 2018 — Fall Armyworm in Africa: A Guide for Integrated Pest Management",
            "TAAT, 2019 — Fall Armyworm Compendium for Extension Workers",
        ],
    },
    "fall-armyworm-larva": {
        "severity": "high",
        "alert_color": "red",
        "display_name": "Fall Armyworm — Active Larvae",
        "description": (
            "Live fall armyworm larvae confirmed. This is the most damaging "
            "stage — larvae feed on leaf tissue, bore into whorls and ears, "
            "and can destroy a crop within days if left uncontrolled."
        ),
        "cultural_control": [
            "Handpick and destroy larvae from whorls — most effective for smallholder farms with low infestation.",
            "Apply fine sand, soil, or ash mixed with a pinch of salt into leaf whorls to kill early-instar larvae.",
            "Crush larvae found during scouting; inspect every plant in infested rows.",
        ],
        "biological_control": [
            "Apply Bacillus thuringiensis var. kurstaki (Bt) suspension directly into whorls — effective against 1st to 3rd instar.",
            "Spray neem seed extract (50 g neem powder per litre of water) — acts as antifeedant and growth regulator.",
            "Apply entomopathogenic fungi (Metarhizium anisopliae or Beauveria bassiana) under humid conditions.",
            "Conserve natural enemies: ants, earwigs, spiders, and parasitoid wasps (Cotesia icipe) actively prey on larvae.",
        ],
        "chemical_control": [
            "Chemical control is justified when >20% of plants have larvae in the whorl.",
            "First choice: chlorantraniliprole (Coragen) at 60 mL/ha — low toxicity to natural enemies.",
            "Alternative: emamectin benzoate at 200 mL/ha or spinetoram at 100 mL/ha.",
            "Apply as a directed whorl spray in late afternoon when larvae are most active.",
            "CRITICAL: Rotate IRAC mode-of-action groups across applications to prevent resistance.",
            "Do NOT use organophosphates or pyrethroids as sole treatment — resistance is widespread.",
        ],
        "prevention": [
            "Adopt push-pull technology: plant Desmodium intortum between maize rows and Brachiaria cv. Mulato II as border grass.",
            "Plant early-maturing, FAW-tolerant maize varieties where available.",
            "Practice crop rotation with non-host crops (cassava, sweet potato, beans).",
            "Maintain field hygiene — destroy crop residues and volunteer plants.",
            "Install pheromone traps at field borders to detect moth arrival early.",
        ],
        "sources": [
            "FAO, 2018 — Integrated Management of Fall Armyworm on Maize: A Guide for FFS in Africa",
            "CABI, 2023 — Fall Armyworm Invasive Species Compendium",
            "CIMMYT/IITA, 2018 — Fall Armyworm IPM Guide for Africa",
            "ICIPE, 2019 — Push-Pull Technology for Fall Armyworm Management",
        ],
    },
    "fall-armyworm-larval-damage": {
        "severity": "medium",
        "alert_color": "orange",
        "display_name": "Fall Armyworm — Feeding Damage",
        "description": (
            "Characteristic windowpane feeding damage or ragged holes on leaves. "
            "Damage is present but active larvae may or may not still be on the "
            "plant — inspect whorls to confirm."
        ),
        "cultural_control": [
            "Scout thoroughly: check whorls and ears for live larvae before applying any treatment.",
            "Map damaged areas in the field to target interventions precisely.",
            "If larvae are no longer present, the plant may recover — monitor growth stage and assess yield impact.",
            "Remove and destroy severely damaged plants that will not produce viable ears.",
        ],
        "biological_control": [
            "If live larvae are found, apply Bt (Bacillus thuringiensis) directly to whorls.",
            "Spray neem oil (azadirachtin-based) to deter further feeding if re-infestation is likely.",
            "Preserve beneficial insects — avoid calendar-based spraying that kills natural enemies.",
        ],
        "chemical_control": [
            "Only spray if live larvae are confirmed alongside fresh damage.",
            "Use targeted whorl applications of chlorantraniliprole or emamectin benzoate.",
            "Do NOT spray old damage with no live larvae — this wastes inputs and harms beneficials.",
        ],
        "prevention": [
            "Increase scouting frequency to twice per week during peak season.",
            "Consider replanting if damage occurred before the V6 growth stage and stand loss exceeds 40%.",
            "Implement push-pull and crop rotation for subsequent seasons.",
        ],
        "sources": [
            "FAO, 2018 — Fall Armyworm Management Guide for FFS",
            "CABI, 2023 — Fall Armyworm Datasheet",
        ],
    },
    "healthy-maize": {
        "severity": "none",
        "alert_color": "green",
        "display_name": "Healthy Maize",
        "description": (
            "No pest damage or disease symptoms detected. The plant appears "
            "healthy with normal growth patterns."
        ),
        "cultural_control": [
            "Continue routine scouting every 5–7 days.",
            "Maintain field hygiene — remove weeds that may harbour pest eggs.",
        ],
        "biological_control": [
            "No intervention needed. Maintain habitat for natural enemies (hedgerows, cover crops).",
        ],
        "chemical_control": [
            "No treatment required. Avoid preventive/prophylactic spraying.",
        ],
        "prevention": [
            "Maintain current agronomic practices.",
            "Install pheromone traps to monitor for moth arrival.",
            "Keep records of pest-free observations to compare across seasons.",
        ],
        "sources": [
            "FAO, 2018 — Fall Armyworm Scouting and Monitoring Guidelines",
        ],
    },
    "maize-streak-disease": {
        "severity": "high",
        "alert_color": "red",
        "display_name": "Maize Streak Virus (MSV)",
        "description": (
            "Characteristic pale yellow streaks along leaf veins, caused by "
            "Maize Streak Virus transmitted by leafhoppers (Cicadulina spp.). "
            "Early infection can reduce yield by 30–100%."
        ),
        "cultural_control": [
            "Rogue (remove and destroy) severely infected plants immediately to reduce virus reservoir.",
            "Avoid late planting — peak leafhopper activity coincides with late-season crops.",
            "Eliminate grass weeds around fields that serve as alternative hosts for leafhoppers.",
            "Use reflective mulch to repel leafhoppers from young plants.",
        ],
        "biological_control": [
            "Encourage natural predators of leafhoppers: ladybird beetles, lacewings, and spiders.",
            "No direct biological control agent is effective against the virus itself.",
        ],
        "chemical_control": [
            "Seed treatment with imidacloprid or thiamethoxam provides early-season leafhopper protection.",
            "Foliar application of acetamiprid can reduce leafhopper populations in severe outbreaks.",
            "Chemical control of the virus itself is not possible — only the vector can be targeted.",
        ],
        "prevention": [
            "Plant MSV-resistant or tolerant maize varieties (e.g., IITA/CIMMYT resistant lines).",
            "Synchronise planting dates within the community to reduce continuous host availability.",
            "Rotate with non-host crops to break the leafhopper breeding cycle.",
            "Scout for leafhoppers early — action threshold is 2+ per plant at seedling stage.",
        ],
        "sources": [
            "CIMMYT, 2020 — Maize Streak Virus Management in Sub-Saharan Africa",
            "IITA, 2019 — Maize Diseases: Identification and Management Guide",
            "CABI, 2023 — Maize Streak Virus Datasheet",
        ],
    },
}


def get_recommendation_details(
    class_name: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> dict:
    """Return structured management guidance for a detected class.

    Includes region-specific advisory when GPS coordinates are available.
    """
    protocol = MANAGEMENT_PROTOCOLS.get(class_name)
    if protocol is None:
        protocol = {
            "severity": "unknown",
            "alert_color": "amber",
            "display_name": class_name.replace("-", " ").title(),
            "description": "Unrecognised detection. Inspect the plant manually.",
            "cultural_control": ["Review the detection manually before acting."],
            "biological_control": ["Consult a local extension officer."],
            "chemical_control": ["Do not apply chemicals without proper identification."],
            "prevention": ["Capture more context images to refine the recommendation."],
            "sources": [],
        }

    # Resolve regional advisory
    region_key = _resolve_region(latitude, longitude)
    region = REGION_ADVISORY[region_key]

    return {
        **protocol,
        "region_advisory": region,
    }


def format_recommendation(class_name: str) -> str:
    """Return a compact human-readable recommendation string."""
    protocol = MANAGEMENT_PROTOCOLS.get(class_name, {})
    severity = protocol.get("severity", "unknown")
    display = protocol.get("display_name", class_name)
    cultural = protocol.get("cultural_control", ["Review manually."])
    return f"{severity.upper()} | {display} — {cultural[0]}"
