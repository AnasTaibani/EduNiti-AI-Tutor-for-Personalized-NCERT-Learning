#!/usr/bin/env python3
# data/generate_concepts_seed.py
"""
Generate a canonical concepts_seed.jsonl covering Grades 6-12 for Science, Math, English.
Writes: data/concepts_seed.jsonl
"""

import json
from pathlib import Path
from datetime import datetime

OUT = Path(__file__).resolve().parents[0] / "concepts_seed.jsonl"
now = datetime.utcnow().isoformat() + "Z"

# Compact lists of topics. We aim for >=100 total concepts.
science_topics = [
    # Grade 6
    ("Grade_6","Science", "Food: Sources and Components"),
    ("Grade_6","Science", "Components of Food"),
    ("Grade_6","Science", "Fibre to Fabric"),
    ("Grade_6","Science", "Sorting Materials"),
    ("Grade_6","Science", "Changes Around Us"),
    ("Grade_6","Science", "Bodies and Movement"),
    ("Grade_6","Science", "Light, Shadows and Reflections"),
    ("Grade_6","Science", "Water: Sources & Safety"),
    ("Grade_6","Science", "Air Around Us"),
    ("Grade_6","Science", "Electricity Basics"),
    # Grade 7
    ("Grade_7","Science", "Nutrition in Plants"),
    ("Grade_7","Science", "Nutrition in Animals"),
    ("Grade_7","Science", "Respiration in Organisms"),
    ("Grade_7","Science", "Transportation in Plants"),
    ("Grade_7","Science", "Heat and Temperature"),
    ("Grade_7","Science", "Acids, Bases & Salts (intro)"),
    ("Grade_7","Science", "Motion and Measurement"),
    ("Grade_7","Science", "Winds and Weather"),
    ("Grade_7","Science", "Soil and Its Composition"),
    ("Grade_7","Science", "Electric Circuits (basic)"),
    # Grade 8
    ("Grade_8","Science", "Microorganisms: Friend and Foe"),
    ("Grade_8","Science", "Crop Production & Management"),
    ("Grade_8","Science", "Synthetic Fibres and Plastics"),
    ("Grade_8","Science", "Conservation of Plants & Animals"),
    ("Grade_8","Science", "Cell: Structure & Function"),
    ("Grade_8","Science", "Force and Pressure"),
    ("Grade_8","Science", "Light: Reflection & Refraction"),
    ("Grade_8","Science", "Reproduction in Plants"),
    ("Grade_8","Science", "Chemical Effects of Current (intro)"),
    ("Grade_8","Science", "Pollution and Its Control"),
    # Grade 9
    ("Grade_9","Science", "Matter: Structure & Properties"),
    ("Grade_9","Science", "Is Matter Around Us Pure?"),
    ("Grade_9","Science", "The Fundamental Unit of Life (Cell)"),
    ("Grade_9","Science", "Tissues & Diversity in Living Organisms"),
    ("Grade_9","Science", "Motion: Laws & Graphs"),
    ("Grade_9","Science", "Force & Laws of Motion"),
    ("Grade_9","Science", "Light: Human Eye & Colours"),
    ("Grade_9","Science", "Natural Resources & Sustainable Use"),
    ("Grade_9","Science", "Food Production and Microbes"),
    ("Grade_9","Science", "Acids, Bases, Salts (expanded)"),
    # Grade 10
    ("Grade_10","Science", "Chemical Reactions & Equations"),
    ("Grade_10","Science", "Acids, Bases & Salts (applications)"),
    ("Grade_10","Science", "Life Processes: Nutrition & Respiration"),
    ("Grade_10","Science", "Control & Coordination"),
    ("Grade_10","Science", "Heredity & Evolution (intro)"),
    ("Grade_10","Science", "Electricity (Ohm's law)"),
    ("Grade_10","Science", "Magnetic Effects of Electric Current"),
    ("Grade_10","Science", "Carbon and its Compounds"),
    ("Grade_10","Science", "Periodic Classification (basic)"),
    ("Grade_10","Science", "Sources of Energy"),
    # Grade 11
    ("Grade_11","Biology", "Biological Classification"),
    ("Grade_11","Biology", "Plant Kingdom"),
    ("Grade_11","Biology", "Photosynthesis"),
    ("Grade_11","Biology", "Human Physiology: Digestive System"),
    ("Grade_11","Biology", "Human Physiology: Circulatory System"),
    ("Grade_11","Chemistry", "Some Basic Concepts of Chemistry"),
    ("Grade_11","Chemistry", "Structure of Atom"),
    ("Grade_11","Physics", "Units and Measurements"),
    ("Grade_11","Physics", "Motion in a Straight Line"),
    ("Grade_11","Physics", "Work, Energy & Power"),
    # Grade 12
    ("Grade_12","Biology", "Biomolecules & Enzymes"),
    ("Grade_12","Biology", "Photosynthesis (advanced)"),
    ("Grade_12","Chemistry", "Organic Chemistry: Hydrocarbons"),
    ("Grade_12","Physics", "Electricity & Circuits (advanced)"),
    ("Grade_12","Physics", "Electromagnetic Induction"),
    ("Grade_12","Physics", "Optics (wave optics basics)"),
    ("Grade_12","Chemistry", "Chemical Kinetics"),
]

math_topics = [
    # Grade 6-8 basics
    ("Grade_6","Mathematics","Integers & Numbers"),
    ("Grade_6","Mathematics","Fractions & Decimals"),
    ("Grade_6","Mathematics","Basic Geometry: Lines & Angles"),
    ("Grade_6","Mathematics","Ratio & Proportion"),
    ("Grade_6","Mathematics","Mensuration: Area & Perimeter"),
    ("Grade_7","Mathematics","Rational Numbers & Operations"),
    ("Grade_7","Mathematics","Algebra: Simple Equations"),
    ("Grade_7","Mathematics","Triangles & Congruence"),
    ("Grade_7","Mathematics","Symmetry & Patterns"),
    ("Grade_7","Mathematics","Data Handling (intro)"),
    ("Grade_8","Mathematics","Linear Equations in One Variable"),
    ("Grade_8","Mathematics","Squares & Cubes"),
    ("Grade_8","Mathematics","Comparing Quantities (percentage)"),
    ("Grade_8","Mathematics","Practical Geometry"),
    ("Grade_8","Mathematics","Introduction to Graphs"),
    # Grade 9
    ("Grade_9","Mathematics","Number Systems (real numbers)"),
    ("Grade_9","Mathematics","Polynomials"),
    ("Grade_9","Mathematics","Coordinate Geometry: Line"),
    ("Grade_9","Mathematics","Statistics: Mean, Median, Mode"),
    ("Grade_9","Mathematics","Introduction to Euclid's Geometry"),
    # Grade 10
    ("Grade_10","Mathematics","Real Numbers & Theorems"),
    ("Grade_10","Mathematics","Triangles: Similarity"),
    ("Grade_10","Mathematics","Trigonometry: ratios"),
    ("Grade_10","Mathematics","Quadratic Equations"),
    ("Grade_10","Mathematics","Circles: Properties"),
    ("Grade_10","Mathematics","Areas related to Circles"),
    # Grade 11
    ("Grade_11","Mathematics","Sets & Functions"),
    ("Grade_11","Mathematics","Limits & Derivatives (intro)"),
    ("Grade_11","Mathematics","Trigonometric Functions"),
    ("Grade_11","Mathematics","Statistics & Probability (intro)"),
    ("Grade_11","Mathematics","Complex Numbers (intro)"),
    # Grade 12
    ("Grade_12","Mathematics","Integration (intro)"),
    ("Grade_12","Mathematics","Applications of Derivatives"),
    ("Grade_12","Mathematics","Matrices & Determinants"),
    ("Grade_12","Mathematics","Probability Distributions"),
    ("Grade_12","Mathematics","Vector Algebra (basic)"),
]

english_topics = [
    ("Grade_6","English","Parts of Speech (basic)"),
    ("Grade_6","English","Comprehension: Short Passages"),
    ("Grade_7","English","Paragraph Writing & Narration"),
    ("Grade_7","English","Tenses: Present & Past"),
    ("Grade_8","English","Active and Passive Voice"),
    ("Grade_8","English","Formal vs Informal Letters"),
    ("Grade_9","English","Poetry: Central Idea & Theme"),
    ("Grade_9","English","Prose: Character & Summary"),
    ("Grade_10","English","Unseen Passage Techniques"),
    ("Grade_10","English","Grammar: Subject-Verb Agreement"),
    ("Grade_11","English","Drama: Understanding Scenes"),
    ("Grade_11","English","Literary Devices: Metaphor, Simile"),
    ("Grade_12","English","Critical Analysis of Poetry"),
    ("Grade_12","English","Essay Writing: Arguments & Structure"),
]

# Combine lists
all_topics = science_topics + math_topics + english_topics

# Ensure uniqueness and stable IDs
concepts = []
used_ids = set()
for (grade, subject, label) in all_topics:
    # create stable, short ID
    base = f"{grade}__{subject}__{label}"
    # simplify to alphanumeric underscores
    cid = base.replace(" ", "_").replace(":", "").replace(",", "").replace("'", "")
    cid = cid.replace("-", "_").replace("&", "and")
    # lower-case stable id
    concept_id = cid
    i = 1
    while concept_id in used_ids:
        i += 1
        concept_id = f"{cid}_{i}"
    used_ids.add(concept_id)

    # example questions (2 per concept) - short templated examples
    example_questions = [
        f"What is/are the key idea(s) of '{label}'?",
        f"Give a simple example or definition for {label}."
    ]

    # parents heuristic: subject-level parent
    parents = [f"{grade}__{subject}"]

    concept = {
        "concept_id": concept_id,
        "label": label,
        "grade": grade,
        "subject": subject,
        "parents": parents,
        "example_questions": example_questions,
        "meta": {
            "seeded_at": now,
            "source": "NCERT-mapping-seed-script"
        }
    }
    concepts.append(concept)

# write JSONL
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for c in concepts:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"Wrote {len(concepts)} concepts to: {OUT}")
