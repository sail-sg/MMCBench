import argparse
import os
import gzip
import torch
import pandas as pd
from tqdm import tqdm
import open_clip
from architecture import ImageGenerator

@torch.no_grad()
def compute_similarities(chunk, model, clip_model, device, clip_tokenizer, clip_preprocess, args):
    """
    Computes the similarities between images and captions.

    :param chunk: DataFrame chunk containing captions and keys.
    :param model: The model used for generating clean and corrupted images.
    :param clip_model: CLIP model for feature encoding.
    :param device: Device to run the computations on.
    :param clip_tokenizer: Tokenizer for the CLIP model.
    :param clip_preprocess: Preprocessing function for images.
    :param args: Command line arguments.
    :return: Tuple of dictionaries containing similarities and image data.
    """
    keys = chunk['key'].tolist()
    captions = chunk['caption'].tolist()

    clean_imgs = model(captions=captions)
    image_features = clip_model.encode_image(torch.stack([clip_preprocess(img) for img in clean_imgs]).to(device))
    text_features = clip_model.encode_text(clip_tokenizer(captions).to(device))

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = {'cross_similarity': pd.DataFrame(), 'uni_similarity': pd.DataFrame()}
    image_df = pd.DataFrame({'key': keys, 'caption': captions})

    if args.store_output:
        image_df['clean'] = [img.resize((256, 256)) for img in clean_imgs]

    ## go over each corruption method ##
    for method in tqdm(chunk.columns.difference(['key', 'caption'])):
        corrupted_captions = chunk[method].tolist()
        corrupted_imgs = model(captions=corrupted_captions)
        corrupted_image_features = clip_model.encode_image(torch.stack([clip_preprocess(img) for img in corrupted_imgs]).to(device))
        corrupted_image_features /= corrupted_image_features.norm(dim=-1, keepdim=True)

        similarities['cross_similarity'][method] = (image_features * corrupted_image_features).sum(1).detach().cpu().numpy()
        similarities['uni_similarity'][method] = (text_features * corrupted_image_features).sum(1).detach().cpu().numpy()

        if args.store_output:
            image_df[method] = [img.resize((256, 256)) for img in corrupted_imgs]

    return similarities, image_df

def main(args):
    file = f"data/{args.degree}_corrupted_{args.level}.csv"
    chunks = pd.read_csv(file, chunksize=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImageGenerator(args.model, device = device)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    clip_model.to(device).eval()
    clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')

    all_similarities = {'cross_similarity': pd.DataFrame(), 'uni_similarity': pd.DataFrame()}
    images_list = []

    for chunk in tqdm(chunks):
        chunk_similarities, chunk_images = compute_similarities(chunk, model, clip_model, device, clip_tokenizer, clip_preprocess, args)
        for key in all_similarities.keys():
            all_similarities[key] = pd.concat([all_similarities[key], chunk_similarities[key]], ignore_index=True)
        images_list.append(chunk_images)
        
    all_images = pd.concat(images_list, ignore_index=True)

    directory = f"results/{args.model}"
    os.makedirs(directory, exist_ok=True)

    if args.store_output:
        image_filename = f'{directory}/{args.degree}_corrupted_{args.level}.pkl.gz'
        with gzip.open(image_filename, 'wb') as f:
            all_images.to_pickle(f, compression='gzip')

    for key in all_similarities.keys():
        filename = f'{directory}/{args.degree}_corrupted_{args.level}_{key}.csv'
        all_similarities[key].to_csv(filename, index=False)

    # Print the results
    print(f"Model: {args.model}, Dataset level: {args.level}, Corruption degree: {args.degree}")
    print('Cross similarity: ', f"{((all_similarities['cross_similarity'].values[:,1:]).sum(1).mean()*100):.0f}")
    print('Uni similarity: ', f"{((all_similarities['uni_similarity'].values[:,1:]).sum(1).mean()*100):.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for computing image-text similarities.")
    parser.add_argument("--level", type=str, default='hard_1k', help="Specify the dataset level.")
    parser.add_argument("--degree", type=str, default='heavy', help="Specify the corruption degree.")
    parser.add_argument("--model", type=str, default='stable-diffusion-v1-5', help="Model name.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--store_output", action='store_true', help="Flag to store output images.")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    main(args)
