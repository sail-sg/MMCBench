import argparse
import os
import gzip
import pandas as pd
import torch
import open_clip
from tqdm import tqdm
from architecture import CaptionGenerator
import pyarrow.parquet as pq
from utils import byte_array_to_image

def compute_similarities(chunk, caption_generator, clip_model, device, clip_tokenizer, clip_preprocess, args):
    """
    Computes similarities for a given chunk of data.

    :param chunk: DataFrame chunk with images and captions.
    :param caption_generator: Caption generation model.
    :param clip_model: CLIP model for feature encoding.
    :param device: Computation device (CPU/GPU).
    :param clip_tokenizer: Tokenizer for the CLIP model.
    :param clip_preprocess: Preprocessing function for images.
    :param store_output: Boolean to determine if output is stored.
    :return: Tuple of dictionaries containing similarities and image data.
    """
    captions = chunk['caption'].tolist()
    keys = chunk['key'].tolist()

    clean_imgs = chunk['clean'].apply(byte_array_to_image).tolist()
    clean_generated_text = caption_generator(clean_imgs)

    image_features = clip_model.encode_image(torch.stack([clip_preprocess(img) for img in clean_imgs]).to(device))
    text_features = clip_model.encode_text(clip_tokenizer(clean_generated_text).to(device))
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = {'cross_similarity': pd.DataFrame({'key': keys}), 'uni_similarity': pd.DataFrame({'key': keys})}
    text_df = pd.DataFrame({'key': keys, 'caption': captions})

    if args.store_output:
        text_df['clean'] = clean_generated_text

    for method in tqdm(chunk.columns.difference(['key', 'caption', 'clean'])):
        corrupted_imgs = chunk[method].apply(byte_array_to_image).tolist()
        corrupted_generated_text = caption_generator(corrupted_imgs)
        corrupted_text_features = clip_model.encode_text(clip_tokenizer(corrupted_generated_text).to(device))
        corrupted_text_features /= corrupted_text_features.norm(dim=-1, keepdim=True)

        similarities['cross_similarity'][method] = (image_features * corrupted_text_features).sum(1).detach().cpu().numpy()
        similarities['uni_similarity'][method] = (text_features * corrupted_text_features).sum(1).detach().cpu().numpy()

        if args.store_output:
            text_df[method] = corrupted_generated_text

    return similarities, text_df

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    clip_model.to(device).eval()
    clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')
    caption_generator = CaptionGenerator(args.model, device)

    parquet_file = pq.ParquetFile(f'data/{args.degree}_corrupted_{args.level}.parquet')
    all_similarities = {'cross_similarity': pd.DataFrame(), 'uni_similarity': pd.DataFrame()}
    texts_list = []

    for batch in parquet_file.iter_batches(args.batch_size):
        chunk = batch.to_pandas()
        chunk_similarities, chunk_texts = compute_similarities(chunk, caption_generator, clip_model, device, clip_tokenizer, clip_preprocess, args)
        for key in all_similarities.keys():
            all_similarities[key] = pd.concat([all_similarities[key], chunk_similarities[key]], ignore_index=True)
        texts_list.append(chunk_texts)
        break
        
    all_texts = pd.concat(texts_list, ignore_index=True)

    # Save results
    directory = f"results/{args.model}"
    os.makedirs(directory, exist_ok=True)

    if args.store_output:
        text_filename = f'{directory}/{args.degree}_corrupted_{args.level}.pkl.gz'
        with gzip.open(text_filename, 'wb') as f:
            all_texts.to_pickle(f, compression='gzip')

    for key in all_similarities.keys():
        filename = f'{directory}/{args.degree}_corrupted_{args.level}_{key}.csv'
        all_similarities[key].to_csv(filename, index=False)

    # Print the results
    print(f"Model: {args.model}, Dataset level: {args.level}, Corruption degree: {args.degree}")
    print('Cross similarity: ', f"{((all_similarities['cross_similarity'].values[:,1:]).sum(1).mean()*100):.0f}")
    print('Uni similarity: ', f"{((all_similarities['uni_similarity'].values[:,1:]).sum(1).mean()*100):.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for computing image-to-text similarities.")
    parser.add_argument("--model", type=str, default='blip_base', help="Model name for caption generation.")
    parser.add_argument("--level", type=str, default='hard_1k', help="Dataset level.")
    parser.add_argument("--degree", type=str, default='heavy', help="Specify the corruption degree.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--store_output", action='store_true', help="Flag to store output data.")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    main(args)
