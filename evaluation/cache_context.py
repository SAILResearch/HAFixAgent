"""
Pre-extraction script for blame contexts with parallel processing.
Generates cached data for fast agent runtime loading.
"""

import json
import time
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset.defects4j.defects4j_extractor import Defects4JExtractor
from dataset.defects4j.util import ensure_defects4j_docker_container
from hafix_agent.blame.extraction_config import (
    get_bug_info_path, get_blame_info_path, get_bug_info_filename,
    get_blame_info_filename, validate_extraction_params, get_cache_path
)
from hafix_agent.blame.context_loader import create_context_loader

# Load config once at module level to avoid repeated I/O
_CONFIG_CACHE = None

def get_config() -> Dict:
    """Get cached config or load it if not cached."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        config_path = Path(__file__).parent.parent / "config" / "defects4j.yaml"
        _CONFIG_CACHE = yaml.safe_load(config_path.read_text())
    return _CONFIG_CACHE


def extract_complete_bug_worker(args_tuple: Tuple[str, str, str, str, int, str, bool]) -> Dict[str, Any]:
    """Worker function for complete bug extraction (bug info + blame context with single container)."""
    project_name, bug_id, dataset_path, selector_type, n_lines, bug_filter, include_blameless = args_tuple

    shared_docker_env = None
    try:
        # Create fresh extractor instance and runtime context loader
        extractor = Defects4JExtractor()
        runtime_loader = create_context_loader('runtime')

        # Load config for Docker parameters
        config = get_config()
        env_config = config.get('environment', {})

        # Create shared Docker container using utility function (handles checkout automatically)
        shared_docker_env, error = ensure_defects4j_docker_container(
            project_name, bug_id, None,
            image=env_config.get('image'),
            use_existing_container=env_config.get('use_existing_container'),
            cleanup_on_exit=env_config.get('cleanup_on_exit')
        )

        if error:
            return {
                'project_name': project_name,
                'bug_id': bug_id,
                'status': 'failed',
                'error': f"Docker setup failed: {error}",
                'operations': []
            }

        results = {
            'project_name': project_name,
            'bug_id': bug_id,
            'status': 'success',
            'operations': []
        }

        # Always extract bug info (required for blame context)
        try:
            bug_info_result = runtime_loader.get_bug_info(
                project_name=project_name,
                bug_id=bug_id,
                extractor=extractor,
                docker_env=shared_docker_env
            )

            if 'error' not in bug_info_result:
                # Save bug info to cache (re-extraction acceptable for different selector/n_lines)
                bug_info_path = get_bug_info_path(dataset_path, bug_filter)
                bug_info_path.mkdir(parents=True, exist_ok=True)

                filename = get_bug_info_filename(project_name, bug_id)
                file_path = bug_info_path / filename

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(bug_info_result, f, indent=2, default=str)

                results['operations'].append({
                    'type': 'bug_info',
                    'status': 'success',
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size
                })
            else:
                results['operations'].append({
                    'type': 'bug_info',
                    'status': 'failed',
                    'error': bug_info_result['error']
                })
                bug_info_result = None  # Don't pass failed result to blame context

        except Exception as e:
            results['operations'].append({
                'type': 'bug_info',
                'status': 'exception',
                'error': str(e)
            })
            bug_info_result = None

        # Always extract blame context (with specific selector/n_lines combination)
        try:
            blame_result = runtime_loader.get_blame_context(
                project_name=project_name,
                bug_id=bug_id,
                extractor=extractor,
                selector_type=selector_type,
                n_lines=n_lines,
                bug_info=bug_info_result,
                include_blameless=include_blameless,
                docker_env=shared_docker_env,
                model_config=config.get('model', {})  # For LLMJudgeSelector
            )

            # Prepare result data
            result_data = {
                'project_name': project_name,
                'bug_id': bug_id,
                'selector_type': selector_type,
                'n_lines': n_lines,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'blame_info': blame_result.get('blame_info') if 'error' not in blame_result else None
            }

            if 'error' in blame_result:
                result_data['error'] = blame_result['error']

            # Save blame context to cache (different selector/n_lines won't overwrite each other)
            blame_info_path = get_blame_info_path(dataset_path, bug_filter)
            blame_info_path.mkdir(parents=True, exist_ok=True)

            filename = get_blame_info_filename(project_name, bug_id, selector_type, n_lines)
            file_path = blame_info_path / filename

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, default=str)

            results['operations'].append({
                'type': 'blame_context',
                'selector_type': selector_type,
                'n_lines': n_lines,
                'status': 'success' if 'error' not in blame_result else 'failed',
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'error': blame_result.get('error') if 'error' in blame_result else None
            })

        except Exception as e:
            results['operations'].append({
                'type': 'blame_context',
                'selector_type': selector_type,
                'n_lines': n_lines,
                'status': 'exception',
                'error': str(e)
            })

        # Container cleanup handled in finally block via force_cleanup
        return results

    except Exception as e:
        return {
            'project_name': project_name,
            'bug_id': bug_id,
            'status': 'exception',
            'error': str(e),
            'operations': []
        }
    finally:
        if shared_docker_env is not None:
            try:
                shared_docker_env.force_cleanup()
            except Exception as cleanup_error:
                print(f"Warning: force cleanup failed for {project_name}_{bug_id}: {cleanup_error}")




def main():
    """Main function for pre-extracting blame contexts."""
    parser = argparse.ArgumentParser(description="Pre-extract blame contexts for fast agent runtime")
    parser.add_argument("--bug-category", default="single_file_multi_hunk",
                       choices=["all", "single_line", "single_hunk", "single_file_multi_hunk", "multi_file_multi_hunk"],
                       help="Bug category filter")
    parser.add_argument("--project-filter", default="", help="Specific project name (e.g., Math, Lang)")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of bugs to process (0 = no limit)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--dataset-path", default="dataset/defects4j", help="Dataset base path")
    parser.add_argument("--selector-type", default="first", help="Selector type to extract (e.g., first, random, llm_judge)")
    parser.add_argument("--n-lines", type=int, default=1, help="Number of lines to extract")
    parser.add_argument("--cache-mode", default="both",
                       choices=["blameable", "blameless", "both"],
                       help="Which bugs to cache: blameable only, blameless only, or both")

    args = parser.parse_args()

    # Get bugs to process based on cache mode
    extractor = Defects4JExtractor()

    if args.cache_mode in ["blameless", "both"]:
        # Get both blameable and blameless bugs separately
        blameable_bugs, blameless_bugs = extractor.get_filtered_bugs_with_blameless(
            bug_filter=args.bug_category,
            project_filter=args.project_filter,
            limit=args.limit,
            include_blameless=True
        )

        if args.cache_mode == "blameless":
            bugs_to_process = blameless_bugs
            print(f"Cache mode: blameless only - {len(blameless_bugs)} bugs")
        elif args.cache_mode == "both":
            bugs_to_process = blameable_bugs + blameless_bugs
            print(f"Cache mode: both - {len(blameable_bugs)} blameable + {len(blameless_bugs)} blameless bugs")
    else:  # args.cache_mode == "blameable"
        # Use the original method with only blameable bugs
        bugs_to_process = extractor.get_filtered_bugs(
            bug_filter=args.bug_category,
            project_filter=args.project_filter,
            limit=args.limit,
            blame_suitable_only=True
        )
        print(f"Cache mode: blameable only - {len(bugs_to_process)} bugs")

    print(f"Found {len(bugs_to_process)} bugs to process")
    print(f"Using {args.workers} parallel workers")

    # Create dataset cache directory
    cache_path = get_cache_path(args.dataset_path, args.bug_category)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Validate parameters
    validate_extraction_params(args.selector_type, args.n_lines)

    # Process each bug completely (always extract both bug info + blame context)
    print(f"\n=== Processing Complete Bug Extraction ===")
    print(f"Always extracting: bug info + blame context")
    print(f"Selector type: {args.selector_type}")
    print(f"N-lines: {args.n_lines}")

    # Determine if blameless support is needed for extraction
    include_blameless = args.cache_mode in ["blameless", "both"]

    # Create tasks for each bug
    complete_bug_tasks = [
        (project, bug_id, args.dataset_path, args.selector_type, args.n_lines, args.bug_category, include_blameless)
        for project, bug_id in bugs_to_process
    ]

    print(f"Total bugs to process: {len(complete_bug_tasks)}")

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(extract_complete_bug_worker, task): task for task in complete_bug_tasks}

        completed = 0
        for future in as_completed(future_to_task):
            result = future.result()
            completed += 1

            project_name = result['project_name']
            bug_id = result['bug_id']

            if result['status'] == 'success':
                operations_summary = []
                for op in result['operations']:
                    if op['status'] == 'success':
                        if op['type'] == 'bug_info':
                            operations_summary.append(f"bug_info({op['file_size']}B)")
                        else:
                            operations_summary.append(f"blame({op['selector_type']},{op['n_lines']},{op['file_size']}B)")
                    else:
                        operations_summary.append(f"{op['type']}(FAILED)")

                ops_str = ", ".join(operations_summary)
                print(f"[{completed}/{len(complete_bug_tasks)}] ✓ {project_name}_{bug_id} - {ops_str}")
            else:
                print(f"[{completed}/{len(complete_bug_tasks)}] ✗ {project_name}_{bug_id} - {result.get('error', 'Unknown error')}")

    elapsed = time.time() - start_time
    print(f"Complete bug extraction completed in {elapsed:.1f}s")

    print(f"\n✓ Pre-extraction completed!")
    print(f"Cache location: {cache_path}")


if __name__ == "__main__":
    main()
