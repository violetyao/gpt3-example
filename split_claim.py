"""
GPT-3 converse with itself continuing from partial datasets
"""

from typing import List
import argparse
from neural_worker import NeuralWorker
from tqdm import tqdm


def write_splitted_claims_to_file(history, output_file):
    for item in history:
        output_file.write('=====\n')
        output_file.write(item + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_prompt_template_file', type=str, required=True,
                        help='The path to the file containing the GPT-3 prompt.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Where to read the partial conversations from.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Where to write the outputs.')
    parser.add_argument('--engine', type=str, required=True,
                        choices=['ada',
                                 'text-ada-001',
                                 'babbage',
                                 'text-babbage-001',
                                 'curie',
                                 'text-curie-001',
                                 'davinci',
                                 'text-davinci-001',
                                 'text-davinci-002'],
                        help='The GPT-3 engine to use.')  # choices are from the smallest to the largest model

    parser.add_argument('--num_inputs', type=int, default=1, required=False, help='Number of dialogs to read from the input file (default: 1')

    # GPT-3 generation hyperparameters
    parser.add_argument('--max_tokens', type=int, default=40, required=False, help='')
    parser.add_argument('--temperature', type=float, default=0.8, required=False, help='')
    parser.add_argument('--top_p', type=float, default=0.9, required=False, help='')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, required=False, help='')
    parser.add_argument('--presence_penalty', type=float, default=0.0, required=False, help='')
    parser.add_argument('--stop_tokens', nargs='+', type=str,
                        default=None, required=False, help='Stop tokens for generation')

    args = parser.parse_args()


    all_original_responses = []
    with open(args.input_file) as input_file:
        for line in input_file:
            if len(all_original_responses) < args.num_inputs:
                all_original_responses.append(line.strip())
            else:
                break

    # initialize the NeuralWorkers
    generator_neural_worker = NeuralWorker(prompt_template_file=args.generation_prompt_template_file, engine=args.engine)

    splitted_claims_lst = []
    for original_response in all_original_responses:
        filled_prompt = generator_neural_worker.fill_prompt_template(history=original_response)
        reply = generator_neural_worker.generate(input_text=filled_prompt, args=args, postprocess=False, max_tries=1)
        splitted_claims_lst.append(reply)

    with open(args.output_file, 'w') as output_file:
        write_splitted_claims_to_file(splitted_claims_lst, output_file)
