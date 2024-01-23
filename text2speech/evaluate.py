import argparse
import os
import gzip
import pandas as pd
import torch
from tqdm import tqdm
from architecture import SpeechSynthesizer
from utils import SpeechContentSimilarity

def compute_similarities(chunk, speech_generator, uni_model, args):
    """
    Computes the uni similarities for text-to-speech.

    Note: Currently, only uni similarity is calculated. Cross similarity will be added soon.

    :param chunk: DataFrame chunk containing paths and transcriptions.
    :param speech_generator: The TTS model for generating speech.
    :param uni_model: Model to compute the uni similarity.
    :param args: Command line arguments.
    :return: Tuple of dictionaries containing similarities and speech data.
    """
    paths = chunk['path'].tolist()
    transcription = chunk['transcription'].tolist()
    clean_speech = speech_generator(transcription)

    similarities = {'uni_similarity': pd.DataFrame({'path': paths})}
    speech_df = pd.DataFrame({'path': paths, 'transcription': transcription})

    if args.store_output:
        speech_df['clean'] = clean_speech

    for method in tqdm(chunk.columns.difference(['path', 'transcription'])):
        corrupted_text = chunk[method].tolist()
        corrupted_speech = speech_generator(corrupted_text)
        similarities['uni_similarity'][method] = uni_model(clean_speech, corrupted_speech)

        if args.store_output:
            speech_df[method] = corrupted_speech

    return similarities, speech_df

def main(args):
    """
    Main function to process data and compute text-to-speech similarities.

    :param args: Command line arguments.
    """
    file_path = f'data/{args.degree}_corrupted_{args.level}.csv'
    chunks = pd.read_csv(file_path, chunksize=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    speech_generator = SpeechSynthesizer(args.model, device = device)
    uni_model = SpeechContentSimilarity(device = device)

    ## will update cross similarity later ##
    all_similarities = {'uni_similarity': pd.DataFrame()}
    speech_list = []

    for chunk in chunks:
        chunk_similarities, chunk_speech = compute_similarities(chunk, speech_generator, uni_model, args)
        all_similarities['uni_similarity'] = pd.concat([all_similarities['uni_similarity'], chunk_similarities['uni_similarity']], ignore_index=True)
        speech_list.append(chunk_speech)
        
    all_speeches = pd.concat(speech_list, ignore_index=True)

    # Save results
    directory = f"results/{args.model}"
    os.makedirs(directory, exist_ok=True)

    if args.store_output:
        speech_filename = f'{directory}/{args.level}_corruption_{args.degree}.pkl.gz'
        with gzip.open(speech_filename, 'wb') as f:
            all_speeches.to_pickle(f, compression='gzip')

    similarity_filename = f'{directory}/{args.level}_corruption_{args.degree}_uni_similarity.csv'
    all_similarities['uni_similarity'].to_csv(similarity_filename, index=False)

    # Print the results
    print(f"Model: {args.model}, Dataset level: {args.level}, Corruption degree: {args.degree}")
    print('Uni similarity: ', f"{((all_similarities['uni_similarity'].values[:,1:]).sum(1).mean()*100):.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Speech Similarity Evaluation.")
    parser.add_argument("--model", type=str, default='mms-tts-eng', help="Specify the TTS model name.")
    parser.add_argument("--level", type=str, default='hard_1k', help="Dataset level.")
    parser.add_argument("--degree", type=str, default='heavy', help="Corruption degree.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--store_output", action='store_true', help="Flag to store output speech.")
    args = parser.parse_args()

    # Set GPU
    torch.cuda.set_device(args.gpu)
    main(args)
