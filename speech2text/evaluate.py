import argparse
import os
import gzip
import pandas as pd
import torch
import pyarrow.parquet as pq
from tqdm import tqdm
from utils import SentenceSimilarityCalculator, bytes_to_speech
from architecture import TranscriptionGenerator

def compute_similarities(chunk, transcription_generator, sentence_similarity_calculator, args):
    """
    Computes the similarities between transcriptions of speech files.

    :param chunk: DataFrame chunk containing speech paths and transcriptions.
    :param transcription_generator: Model used for generating transcriptions.
    :param sentence_similarity_calculator: Tool to calculate sentence similarity.
    :param args: Command line arguments.
    :return: Tuple of dictionaries containing similarities and text data.
    """
    paths = chunk['path'].tolist()
    transcriptions = chunk['transcription'].tolist()
    clean_speech = chunk['clean'].apply(bytes_to_speech).tolist()
    clean_generated_text = transcription_generator(clean_speech)

    similarities = {'uni_similarity': pd.DataFrame({'path': paths})}
    text_df = pd.DataFrame({'path': paths, 'transcription': transcriptions})

    if args.store_output:
        text_df['clean'] = clean_generated_text

    for method in tqdm(chunk.columns.difference(['path', 'transcription', 'clean'])):
        corrupted_speech = chunk[method].apply(bytes_to_speech).tolist()
        corrupted_generated_text = transcription_generator(corrupted_speech)
        similarities['uni_similarity'][method] = sentence_similarity_calculator(clean_generated_text, corrupted_generated_text)
        
        if args.store_output:
            text_df[method] = corrupted_generated_text

    return similarities, text_df

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentence_similarity_calculator = SentenceSimilarityCalculator(device = device)
    transcription_generator = TranscriptionGenerator(args.model, device = device)

    parquet_file = pq.ParquetFile(f'data/{args.degree}_corrupted_{args.level}.parquet')
    all_similarities = {'uni_similarity': pd.DataFrame()}
    texts_list = []

    for batch in parquet_file.iter_batches(args.batch_size):
        ## the speech here is with 16khz
        chunk = batch.to_pandas()
        chunk_similarities, chunk_text = compute_similarities(chunk, transcription_generator, sentence_similarity_calculator, args)
        all_similarities['uni_similarity'] = pd.concat([all_similarities['uni_similarity'], chunk_similarities['uni_similarity']], ignore_index=True)
        texts_list.append(chunk_text)
        
    all_texts = pd.concat(texts_list, ignore_index=True)

    # Save results
    directory = f"results/{args.model}"
    os.makedirs(directory, exist_ok=True)

    if args.store_output:
        text_filename = f'{directory}/{args.degree}_corrupted_{args.level}.pkl.gz'
        with gzip.open(text_filename, 'wb') as f:
            all_texts.to_pickle(f, compression='gzip')

    similarity_filename = f'{directory}/{args.level}_corruption_{args.degree}_uni_similarity.csv'
    all_similarities['uni_similarity'].to_csv(similarity_filename, index=False)

    # Print the results
    print(f"Model: {args.model}, Dataset level: {args.level}, Corruption degree: {args.degree}")
    print('Uni similarity: ', f"{((all_similarities['uni_similarity'].values[:,1:]).sum(1).mean()*100):.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for evaluating speech-to-text models.")
    parser.add_argument("--model", type=str, default='wav2vec2-base-960h', help="Model name.")
    parser.add_argument("--level", type=str, default='hard_1k', help="Specify the dataset level.")
    parser.add_argument("--degree", type=str, default='heavy', help="Specify the corruption degree.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--store_output", action='store_true', help="Flag to store output speech.")
    args = parser.parse_args()

    # Set GPU
    torch.cuda.set_device(args.gpu)
    main(args)