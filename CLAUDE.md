# ARC-AGI Model Baseline Development Guide

## Commands
- Run single task: `python3 -m main --data_dir data/arc-agi/data/evaluation --provider anthropic --model claude-3-5-sonnet-20241022 --task_id 0a1d4ef5 --print_logs`
- Run batch tasks: `parallel --jobs 20 --progress python3 -m main --data_dir data/arc-agi/data/evaluation --provider anthropic --model claude-3-5-sonnet-20241022 --task_id {} --save_submission_dir submissions/claude_sonnet_20241022 --print_logs :::: ./data/task_lists/public_evaluation.txt`
- Generate task lists: `python3 -m src.utils.generate_tasks_list --task_dir data/arc-agi/data/training --output_file data/task_lists/public_training`
- Score submissions: `python3 -m src.scoring.scoring --task_dir data/arc-agi/data/evaluation --submission_dir submissions/claude_sonnet_20241022 --print_logs --results_dir results/claude_sonnet_20241022`

## Code Style Guide
- Imports: Group standard library, third-party, and local imports
- Types: Use type hints consistently (List, Dict, Optional)
- Naming: PascalCase for classes, snake_case for functions/variables
- Error handling: Use specific exceptions with informative messages
- Models: Use Pydantic for data models
- Documentation: Include docstrings for classes and methods
- Implementation: Follow adapter pattern for provider implementations