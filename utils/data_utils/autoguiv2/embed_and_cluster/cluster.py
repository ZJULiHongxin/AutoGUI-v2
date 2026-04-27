import os, json, math, argparse, glob
from typing import Dict, List, Tuple, Any

import numpy as np
from tqdm import tqdm

try:
    from sklearn.cluster import KMeans
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required for clustering. Please install it: pip install scikit-learn"
    ) from e


def load_embedding_from_file(path: str, field: str, context: str) -> np.ndarray:
    with open(path, 'r') as f:
        obj = json.load(f)
    try:
        vec = obj[field][context]
    except KeyError as ke:
        raise KeyError(
            f"Embedding file {path} does not contain requested key path '{field}/{context}'."
        ) from ke
    return np.asarray(vec, dtype=np.float32)


def find_embedding_files(root: str) -> List[Tuple[str, str, str]]:
    files = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Embeddings root directory not found: {root}")
    for img_name in sorted(os.listdir(root)):
        img_dir = os.path.join(root, img_name)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith('.json'):
                continue
            node_id = os.path.splitext(fname)[0]
            files.append((img_name, node_id, os.path.join(img_dir, fname)))
    if len(files) == 0:
        raise RuntimeError(f"No embedding files (*.json) found under {root}")
    return files


def compute_kmeans_centers(
    X: np.ndarray, n_clusters: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be > 0")
    if X.shape[0] < n_clusters:
        n_clusters = X.shape[0]
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    return labels, centers


def select_medoids_from_centroids(
    X: np.ndarray, labels: np.ndarray, centers: np.ndarray
) -> List[int]:
    medoid_indices: List[int] = []
    for k in range(centers.shape[0]):
        members = np.where(labels == k)[0]
        if members.size == 0:
            continue
        diffs = X[members] - centers[k]
        dists = np.einsum('ij,ij->i', diffs, diffs)  # squared L2
        medoid_indices.append(members[int(np.argmin(dists))])
    return medoid_indices


def cluster_global(
    cache_dir: str,
    emb_root: str,
    field: str,
    context: str,
    n_clusters: int,
    ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str, str, int]], Dict[str, Any]]:
    triplets, processed_triplets = find_embedding_files(emb_root), []
    X_list: List[np.ndarray] = []
    for img_name, node_id, fpath in tqdm(triplets, desc="Loading embeddings"):
        # Only process the fixed regions
        fix_files = glob.glob(os.path.join(cache_dir, img_name, 'nodes', f'{node_id}_meta_fix*.json'))
        if len(fix_files) == 0: continue

        processed_triplets.append((img_name, node_id, fpath))
        X_list.append(load_embedding_from_file(fpath, field, context))
    X = np.vstack(X_list)

    if n_clusters is None:
        n_clusters = max(1, int(math.ceil(len(triplets) * ratio)))

    labels, centers = compute_kmeans_centers(X, n_clusters, seed)
    medoid_indices = select_medoids_from_centroids(X, labels, centers)

    # debug
    clusters = {}
    for label, triplet in zip(labels, processed_triplets):
        if label not in clusters:
            clusters[label] = []

        clusters[label].append(triplet)

    centers_records: List[Tuple[str, str, str, int]] = []
    for cluster_id, idx in enumerate(medoid_indices):
        img_name, node_id, fpath = processed_triplets[idx]
        centers_records.append((img_name, node_id, fpath, cluster_id))

    meta = {
        'mode': 'global',
        'total_embeddings': len(processed_triplets),
        'n_clusters': len(medoid_indices),
        'field': field,
        'context': context,
        'embedding_root': emb_root,
        'seed': seed,
    }
    return centers_records, clusters, meta


def filter_source_json(
    source_json_path: str,
    centers: List[Tuple[str, str, str, int]],
) -> Dict[str, Any]:
    with open(source_json_path, 'r') as f:
        src = json.load(f)

    # Try to detect structure like {"results": {img_filename: anno_info, ...}}
    # We will match by image stem (before extension) to the directory name `img_name`.
    results = src.get('results') if isinstance(src, dict) else None
    if not isinstance(results, dict):
        # Fallback: return a compact structure with just identifiers
        return {
            'items': [
                {
                    'img': img,
                    'node_id': node,
                    'cluster_id': cid,
                }
                for (img, node, _path, cid) in centers
            ]
        }

    # Build map from image stem to original filename key
    stem_to_key: Dict[str, str] = {}
    for k in results.keys():
        stem = os.path.splitext(os.path.basename(k))[0]
        stem_to_key[stem] = k

    filtered: Dict[str, Any] = {}
    for img, node_id, _path, cid in centers:
        key = stem_to_key.get(img)
        if key is None:
            continue
        anno = results.get(key)
        if anno is None:
            continue
        if 'result' not in anno or not isinstance(anno['result'], dict):
            continue
        if key not in filtered:
            filtered[key] = {'result': {}}
        node_entry = anno['result'].get(node_id)
        if node_entry is None:
            continue
        filtered[key]['result'][node_id] = node_entry
        filtered[key]['result'][node_id]['cluster_id'] = cid

    return filtered


def save_centers_json(
    output_path: str,
    centers: List[Tuple[str, str, str, int]],
    clusters: List,
    meta: Dict[str, Any],
    source_json_path,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    filtered = filter_source_json(source_json_path, centers)
    out_obj = {
        'meta': meta,
        'results': filtered,
    }

    with open(output_path, 'w') as f:
        print(f"Save clustered functional regions to {output_path}")
        json.dump(out_obj, f, indent=2)

    # Debug
    with open(source_json_path, 'r') as f:
        anno_results = json.load(f)['results']

    mapping = {os.path.basename(k)[:-4]: k for k in anno_results.keys()}

    clustered_functions = []
    for cluster_id, cluster in clusters.items():
        func_cluster = []
        for img_name, node_id, anno_json_file in cluster:
            func_cluster.append(anno_results[mapping[img_name]]['result'][node_id]['functionality']['with_context'])

        clustered_functions.append(func_cluster)

    cluster_funcanno_file = output_path.replace(".json", "_func-cluster.json")
    with open(cluster_funcanno_file, 'w') as f:
        print(f"Save clustered functional descriptions to {cluster_funcanno_file}")
        json.dump(clustered_functions, f, indent=2)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cluster function embeddings and retain cluster centers.")
    ap.add_argument('--embeddings-root', type=str, default="/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/gemini-2.5-pro-thinking/v2/embeddings",
                    help='Root directory containing per-image subfolders of node embedding JSONs.')
    ap.add_argument('--output-json', type=str, default=None,
                    help='Path to write centers JSON.')
    ap.add_argument('--field', type=str, default='functionality', choices=['description', 'functionality'],
                    help='Which field to use from embedding files.')
    ap.add_argument('--context', type=str, default='with_context', choices=['with_context', 'wo_context'],
                    help='Which context to use from embedding files.')
    ap.add_argument('--n-clusters', type=int, default=100,
                    help='Number of clusters. If omitted, computed from --ratio.')
    ap.add_argument('--ratio', type=float, default=0.1,
                    help='If --n-clusters not set, use ceil(ratio * N).')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for KMeans.')
    ap.add_argument('--mode', type=str, default='global', choices=['global', 'per_image'],
                    help='Cluster globally or per image.')
    ap.add_argument('--source-json', type=str, default=None,
                    help='Optional source annotations JSON to filter to centers only.')
    return ap.parse_known_args()


def main() -> None:
    args, _ = parse_args()
    
    parts = args.embeddings_root.split('/')
    bmk_name, model_name, version = parts[-4:-1]
    
    cache_dir = os.path.join('/'.join(parts[:-4]), 'cache', bmk_name, model_name, version)
    source_json = os.path.join(os.path.dirname(args.embeddings_root), 'MMInstruction-OSWorld-G.json')
    output_file = source_json.replace(".json", "_clustered.json") if args.output_json is None else args.output_json

    if args.mode == 'global':
        centers, clusters, meta = cluster_global(
            cache_dir,
            args.embeddings_root,
            args.field,
            args.context,
            args.n_clusters,
            args.ratio,
            args.seed,
        )


    save_centers_json(output_file, centers, clusters, meta, source_json)


if __name__ == '__main__':
    main()

