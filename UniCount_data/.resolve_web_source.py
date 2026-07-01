import json
import os
import sys
import shutil
import io
import tarfile
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download


def collect_tar_files(root: Path):
    """Recursively find all .tar files in directory."""
    return sorted([p for p in root.rglob('*.tar') if p.is_file()])


def normalize_entry(entry: str, repo_id: str) -> str:
    """Normalize manifest entry to filename."""
    entry = entry.strip()
    if not entry:
        return ''
    
    # Full HF URL
    if entry.startswith('http://') or entry.startswith('https://'):
        filename = entry.split('/resolve/main/', 1)[-1].split('?', 1)[0]
        if filename == entry:  # No '/resolve/main/' found
            filename = Path(entry.split('?', 1)[0]).name
        return filename
    
    # Absolute path
    if entry.startswith('/'):
        return entry
    
    # repo_id/filename format
    if '/' in entry and entry.startswith(repo_id):
        return entry.split(repo_id + '/', 1)[-1]
    
    # Plain filename
    return entry


def _caption_from_sidecar(sidecar_path: Path) -> str:
    """Read caption text from .txt/.json sidecar if available."""
    if not sidecar_path.exists() or not sidecar_path.is_file():
        return ''

    if sidecar_path.suffix.lower() == '.txt':
        try:
            return sidecar_path.read_text(encoding='utf-8', errors='replace').strip()
        except Exception:
            return ''

    if sidecar_path.suffix.lower() == '.json':
        try:
            payload = json.loads(sidecar_path.read_text(encoding='utf-8', errors='replace'))
        except Exception:
            return ''

        if isinstance(payload, dict):
            for key in ('caption', 'text', 'description', 'blip_caption', 'alt_text'):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ''

    return ''


def create_webdataset_tar_from_dir(source_dir: Path, out_tar: Path, max_items: int = 0) -> int:
    """Build a synthetic WebDataset tar from extracted files.

    Expected output members are <key>.<image_ext> and <key>.txt.
    """
    image_exts = {'.jpg', '.jpeg', '.png'}
    image_files = sorted(
        p for p in source_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in image_exts
    )

    if not image_files:
        return 0

    out_tar.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(out_tar, 'w') as tar:
        for idx, image_path in enumerate(image_files):
            if max_items > 0 and written >= max_items:
                break

            key = f'sample-{idx:08d}'
            img_ext = image_path.suffix.lower().lstrip('.')
            if img_ext == 'jpeg':
                img_ext = 'jpg'

            image_bytes = image_path.read_bytes()
            image_name = f'{key}.{img_ext}'
            image_info = tarfile.TarInfo(name=image_name)
            image_info.size = len(image_bytes)
            tar.addfile(image_info, io.BytesIO(image_bytes))

            caption = _caption_from_sidecar(image_path.with_suffix('.txt'))
            if not caption:
                caption = _caption_from_sidecar(image_path.with_suffix('.json'))
            caption_bytes = caption.encode('utf-8', errors='replace')

            caption_name = f'{key}.txt'
            caption_info = tarfile.TarInfo(name=caption_name)
            caption_info.size = len(caption_bytes)
            tar.addfile(caption_info, io.BytesIO(caption_bytes))

            written += 1

    return written


def resolve_web_manifest(manifest_path: Path, repo_id: str, cache_root: Path) -> Path:
    """
    Resolve a web manifest to local TAR files.
    
    Process:
    1. Parse manifest file (one entry per line)
    2. Download and/or extract archives (ZIP files)
    3. Collect TAR files
    4. Create symlink directory
    5. Write resolved directory path to file
    """
    downloads_dir = cache_root / 'downloads'
    extracted_dir = cache_root / 'extracted'
    resolved_dir = cache_root / 'resolved_tars'
    linked_dir = resolved_dir / 'linked_tars'
    generated_dir = resolved_dir / 'generated_tars'
    max_items = int(os.environ.get('MAX_SAMPLES', '0') or 0)
    
    # Create directories
    downloads_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    linked_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse manifest
    print(f'[RESOLVER] Reading manifest: {manifest_path}')
    sys.stdout.flush()
    manifest_entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                manifest_entries.append(line)
    
    print(f'[RESOLVER] Found {len(manifest_entries)} entries in manifest')
    sys.stdout.flush()
    
    if not manifest_entries:
        raise ValueError(f'Manifest is empty: {manifest_path}')
    
    # Resolve entries to TAR files
    resolved_paths = []
    
    for idx, entry in enumerate(manifest_entries, 1):
        normalized = normalize_entry(entry, repo_id)
        if not normalized:
            continue
        
        print(f'[RESOLVER] Processing entry {idx}/{len(manifest_entries)}: {normalized}')
        sys.stdout.flush()
        
        # Check if it's a local absolute path
        if normalized.startswith('/') and Path(normalized).exists():
            candidate = Path(normalized)
        else:
            # Download from HF Hub
            candidate_path = hf_hub_download(
                repo_id=repo_id,
                repo_type='dataset',
                filename=normalized,
                cache_dir=str(downloads_dir),
                resume_download=True,
            )
            candidate = Path(candidate_path)
        
        print(f'[RESOLVER]   Downloaded/found: {candidate}')
        sys.stdout.flush()
        
        # Handle TAR files
        if candidate.suffix.lower() == '.tar':
            resolved_paths.append(candidate.resolve())
            continue
        
        # Handle ZIP files
        if candidate.suffix.lower() == '.zip':
            unzip_dir = extracted_dir / candidate.stem
            done_marker = unzip_dir / '.done'
            
            # Extract if not already done
            if not done_marker.exists():
                print(f'[RESOLVER]   Extracting ZIP...')
                sys.stdout.flush()
                unzip_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(candidate) as archive:
                    archive.extractall(unzip_dir)
                done_marker.write_text('done\n')
                print(f'[RESOLVER]   Extraction complete')
            else:
                print(f'[RESOLVER]   Already extracted')
            
            sys.stdout.flush()
            
            # Find TAR files in extracted directory
            tar_files = collect_tar_files(unzip_dir)
            if tar_files:
                print(f'[RESOLVER]   Found {len(tar_files)} TAR files in archive')
                resolved_paths.extend([p.resolve() for p in tar_files])
                sys.stdout.flush()
                continue
            else:
                synthesized_tar = generated_dir / f'{candidate.stem}.tar'
                if not synthesized_tar.exists():
                    print('[RESOLVER]   No TAR in ZIP; synthesizing WebDataset TAR from extracted files...')
                    sys.stdout.flush()
                    written = create_webdataset_tar_from_dir(unzip_dir, synthesized_tar, max_items=max_items)
                    if written == 0:
                        raise RuntimeError(f'No image files found to synthesize TAR from archive: {candidate}')
                    print(f'[RESOLVER]   Synthesized {synthesized_tar} with {written} samples')
                    sys.stdout.flush()
                else:
                    print(f'[RESOLVER]   Reusing synthesized TAR: {synthesized_tar}')
                    sys.stdout.flush()
                resolved_paths.append(synthesized_tar.resolve())
                continue
        
        raise RuntimeError(f'Unsupported archive type: {candidate}')
    
    if not resolved_paths:
        raise ValueError(f'No usable TAR files resolved from manifest')
    
    print(f'[RESOLVER] Total TAR files resolved: {len(resolved_paths)}')
    sys.stdout.flush()
    
    # Write resolved paths to file
    resolved_manifest = resolved_dir / 'resolved_shards.txt'
    resolved_manifest.write_text('\n'.join(str(p) for p in resolved_paths) + '\n')
    print(f'[RESOLVER] Wrote resolved manifest: {resolved_manifest}')
    sys.stdout.flush()
    
    # Create symlinks
    print(f'[RESOLVER] Creating symlinks in {linked_dir}...')
    sys.stdout.flush()
    for idx, shard_path in enumerate(resolved_paths):
        link_name = linked_dir / f'{idx:06d}-{shard_path.name}'
        if not link_name.exists():
            link_name.symlink_to(shard_path)
    
    print(f'[RESOLVER] Created {len(resolved_paths)} symlinks')
    sys.stdout.flush()
    
    # Write resolved directory path
    result_file = cache_root / 'resolved_data_dir.txt'
    result_file.write_text(str(linked_dir) + '\n')
    print(f'[RESOLVER] Wrote resolved data dir to: {result_file}')
    sys.stdout.flush()
    
    return linked_dir


def validate_snapshot(snapshot_dir: Path):
    """Check if HF snapshot is complete."""
    index_file = snapshot_dir / 'model.safetensors.index.json'
    if not index_file.exists():
        return True, []
    
    try:
        index = json.loads(index_file.read_text())
    except Exception:
        return False, ['model.safetensors.index.json (corrupt)']
    
    shard_names = sorted(set(index.get('weight_map', {}).values()))
    missing = [n for n in shard_names if not (snapshot_dir / n).exists()]
    return len(missing) == 0, missing


# Main script
if __name__ == '__main__':
    # Get environment variables
    use_web = os.environ.get('USE_WEB_MANIFEST', '0') == '1'
    data_source = Path(os.environ.get('UNICOUNT_DATA_SOURCE', '.')).expanduser().resolve()
    repo_id = os.environ.get('SA1B_WEB_REPO_ID', 'hdtech/SA-1B')
    cache_root = Path(os.environ.get('SA1B_WEB_WORKDIR', '.sa1b_web_cache')).expanduser().resolve()
    model_path = os.environ.get('REX_OMNI_MODEL_PATH', 'IDEA-Research/Rex-Omni')
    hf_home = Path(os.environ.get('HF_HOME', '.hf_cache'))
    
    print(f'[RESOLVER] Starting resolver')
    print(f'[RESOLVER]   data_source={data_source}')
    print(f'[RESOLVER]   use_web_manifest={use_web}')
    print(f'[RESOLVER]   repo_id={repo_id}')
    print(f'[RESOLVER]   cache_root={cache_root}')
    sys.stdout.flush()
    
    try:
        # Resolve data source
        if use_web:
            resolved_data_dir = resolve_web_manifest(data_source, repo_id, cache_root)
        else:
            # Local directory - just verify it has TAR files
            tar_files = collect_tar_files(data_source)
            if not tar_files:
                raise ValueError(f'No TAR files found in: {data_source}')
            resolved_data_dir = data_source
        
        print(f'[RESOLVER] Data source resolved to: {resolved_data_dir}')
        sys.stdout.flush()
        
        # Ensure model snapshot is complete
        if '/' in model_path and not Path(model_path).exists():
            print(f'[RESOLVER] Checking model snapshot: {model_path}')
            sys.stdout.flush()
            repo_cache = hf_home / 'hub' / ('models--' + model_path.replace('/', '--'))
            
            needs_download = True
            if repo_cache.exists() and (repo_cache / 'snapshots').exists():
                # Check existing snapshots
                for snap in (repo_cache / 'snapshots').iterdir():
                    ok, missing = validate_snapshot(snap)
                    if ok:
                        print(f'[RESOLVER] Found valid snapshot')
                        sys.stdout.flush()
                        needs_download = False
                        break
                    else:
                        print(f'[RESOLVER] Invalid snapshot (missing: {missing[:3]})')
                        sys.stdout.flush()
            
            if needs_download:
                print(f'[RESOLVER] Downloading model snapshot...')
                sys.stdout.flush()
                if repo_cache.exists():
                    shutil.rmtree(repo_cache)
                snapshot_download(
                    repo_id=model_path,
                    cache_dir=str(hf_home),
                    resume_download=True,
                )
                print(f'[RESOLVER] Model snapshot downloaded')
                sys.stdout.flush()
        
        print(f'[RESOLVER] All tasks completed successfully')
        sys.stdout.flush()
        
    except Exception as e:
        print(f'[RESOLVER] ERROR: {e}')
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.exit(1)
