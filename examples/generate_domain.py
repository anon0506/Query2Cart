"""Generate benchmark tasks for a new product domain.

Usage:
    python examples/generate_domain.py
"""

from generation.pipeline import Pipeline

pipeline = Pipeline(
    catalog="path/to/your/catalog.parquet",
    domain="your domain description",
    item_noun="product",
    output_dir="./my_benchmark",
    model_map={
        "TRIAGE": "gpt-5.4",
        "CONFIGURE": "gpt-5.4",
        "COHERENCE": "o4-mini",
        "EXTRACT_DESCRIPTIONS": "gpt-5.4",
        "EXTRACT_DESCRIPTIONS_DISCOVERY": "gpt-5.4",
        "EXTRACT_DESCRIPTIONS_EXTRACTION": "gpt-5.4-nano",
        "EXTRACT_DESCRIPTIONS_NORMALIZATION": "gpt-5.4",
        "GENERATE_TASKS": "gpt-5.4",
    },
)

# Run stage by stage for control:
pipeline.profile()       # Stage 1: analyze columns
pipeline.triage()        # Stage 2: classify roles -> review triage_result.json
pipeline.configure()     # Stage 3: generate config -> review config.json
pipeline.generate_coherence()  # LLM coherence rules -> checkpoints/coherence.py
pipeline.calibrate()     # Stage 4: validate calibration
pipeline.extract_descriptions()

pipeline.generate_tasks(n_tasks=250)  # Stage 5: produce tasks.json

# Or run everything at once:
# pipeline.run()
