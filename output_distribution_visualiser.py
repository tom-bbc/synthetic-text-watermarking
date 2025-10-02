# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import json
import os
from typing import Optional

import numpy as np

from generate_and_visualise import generate_with_probabilities, visualise_probabilities

# --------------------------------------------------------------------------- #
#                              MAIN RUN METHOD                                #
# --------------------------------------------------------------------------- #


def output_distribution_visualiser(
    restore_from_file: Optional[str] = None,
    user_prompt: Optional[str] = None,
    save_model_distribution: bool = False,
):
    # If in test mode use pre-generated model output tokens
    if isinstance(restore_from_file, str) and os.path.isfile(restore_from_file):
        with open(restore_from_file, "r", encoding="utf-8") as f:
            token_probabilities = json.load(f)

        print(f" << * >> Restoring LLM logprobs from file: '{restore_from_file}'")

    elif user_prompt is None:
        print(" -- * -- Error: no prompt or valid 'restore_from_file' file provided.")
        return

    else:
        # Load OpenAI credentials from file
        credentials_filepath = "credentials.json"

        with open(credentials_filepath, "r") as fp:
            credentials = json.load(fp)

        openai_api_key = credentials["openai_api_key"]

        # Generate response from model and return probabilities of top tokens
        _, token_probabilities = generate_with_probabilities(
            openai_api_key, user_prompt, save_model_distribution
        )

    # Visualise token probabilties of output sequence
    data_to_plot = [
        [logprob["probability"] for logprob in token["top_logprobs"]]
        for token in token_probabilities
    ]

    top_token_labels = [
        [f'"{logprob["token"]}"' for logprob in token["top_logprobs"]]
        for token in token_probabilities
    ]
    top_token_labels = np.array(top_token_labels).T

    x_labels = [f'"{token["token"]}"' for token in token_probabilities]
    y_labels = np.arange(1, len(data_to_plot[0]) + 1)

    data_to_plot = np.array(data_to_plot)
    data_to_plot = data_to_plot.T

    visualise_probabilities(data_to_plot, x_labels, y_labels, top_token_labels)


# --------------------------------------------------------------------------- #
#                               CLI ENTRYPOINT                                #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    user_prompt = 'Summarise the following news article in 30 words. Article: \n A report revealed that 253 potential victims of slavery were reported in Hampshire and the Isle of Wight, of which one in four were children. Modern slavery, which includes human trafficking, is the illegal exploitation of people for personal or commercial gain. It can take different forms of slavery, such as domestic or labour exploitation, organ harvesting, EU Status exploitation, and financial, sexual and criminal exploitation. Each year, Hampshire and Isle of Wight Fire and Rescue Authority (HIWFRA), combined by all four authorities, the three unitary councils, and the county council, spends around £99m on making "life safer" in the county and preventing slavery and human trafficking. However, a recent report of the HIWFRA has revealed that by June 2023, there were 253 potential victims identified of modern slavery in Hampshire and the Isle of Wight. Of them, one in four were children. According to the Government\'s UK Annual Report on Modern Slavery, 10,613 potential victims were referred to the National Referral Mechanism in the year ended September 2021. In case any member of the Authority or any of its staff suspects slavery or human trafficking activity either within the community or the organisation, then the concerns will be reported through the Service\'s Safeguarding Reporting Procedure. If slavery or human trafficking activity is suspected through its supply chain, it will be reported to Hampshire Constabulary via the Modern Slavery Helpline. If you need help, advice or information about any modern slavery issue, you can contact the modern slavery helpline confidentially, 24 hours a day, 365 days a year on 0800 012 1700"'
    output_distribution_visualiser(
        user_prompt=user_prompt, save_model_distribution=True
    )

    # pregenerated_tokens_file = "outputs/logprobs/test_20251002.json"
    # output_distribution_visualiser(restore_from_file=pregenerated_tokens_file)
