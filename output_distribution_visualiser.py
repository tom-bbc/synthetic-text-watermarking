# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from openai import AzureOpenAI
from plotly import graph_objects as go

# --------------------------------------------------------------------------- #
#                   GENERATE RESPONSE FROM OPENAI MODEL                       #
# --------------------------------------------------------------------------- #


def generate_with_probabilities(
    openai_api_key: str,
    user_prompt: str,
    save_output_tokens: bool = False,
) -> Tuple[Optional[str], List[Dict]]:
    """
    Generate response to prompt using OpenAI API,
    returning top log probabilities of each token in vocab.
    """

    # Define LLM prompt
    system_prompt = ""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    # Setup connection to OpenAI model
    model_name = "gpt-4.1"
    openai_api_version = "2024-07-01-preview"
    openai_azure_endpoint = "https://ai-research-swedencentral.openai.azure.com"

    openai_client = AzureOpenAI(
        azure_deployment=model_name,
        api_key=openai_api_key,
        api_version=openai_api_version,
        azure_endpoint=openai_azure_endpoint,
    )

    # Define hyperparams
    max_tokens = 1024
    temperature = 1
    logprobs = True
    n_logprobs = 20

    # Send request to OpenAI model API
    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=n_logprobs,
    )

    # Extract content of the response
    response_content = completion.choices[0].message.content

    # Extract the log probabilities of the generated tokens
    log_probabilities = completion.choices[0].logprobs.content
    formatted_probabilities = []

    for token_logprobs in log_probabilities:
        token_logprobs = dict(token_logprobs)
        token_logprobs.pop("bytes")

        top_logprobs = []
        for logprob in token_logprobs["top_logprobs"]:
            logprob = dict(logprob)
            logprob["probability"] = math.exp(logprob["logprob"])
            logprob.pop("bytes")
            top_logprobs.append(logprob)

        token_logprobs["top_logprobs"] = top_logprobs
        formatted_probabilities.append(token_logprobs)

    if save_output_tokens:
        timestamp = datetime.now().strftime("%Y%m%d_%H:%M")
        filename = f"outputs/logprobs/{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(formatted_probabilities, indent=4))

    return response_content, formatted_probabilities


# --------------------------------------------------------------------------- #
#                     PLOT GENERATED TOKEN PROBABILITIES                      #
# --------------------------------------------------------------------------- #


def visualise_probabilities(data, x_labels, y_labels, grid_labels, interactive=True):
    if interactive:
        x_range = np.arange(0, len(x_labels))
        y_range = np.arange(0, len(y_labels))

        fig = go.Figure(
            data=go.Heatmap(
                z=data,
                x=x_range,
                y=y_range,
                colorscale="YlOrBr",
                colorbar=dict(title="Probability"),
                hoverinfo="text",
                text=grid_labels,
            )
        )

        fig.update_layout(
            width=1200,
            height=800,
            xaxis_title="Position in generated sequence",
            yaxis_title="Top 20 candidate output tokens (indexed by probability)",
            xaxis=dict(
                tickmode="array",
                tickangle=-90,
                tickvals=x_range,
                ticktext=x_labels,
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=y_range,
                ticktext=y_labels,
            ),
        )

        # Add outline around cells where the hover text label matches the x axis label
        for y_idx in range(len(grid_labels)):
            for x_idx in range(len(grid_labels[0])):
                if grid_labels[y_idx][x_idx] == x_labels[x_idx]:
                    fig.add_shape(
                        type="rect",
                        x0=x_idx - 0.5,
                        x1=x_idx + 0.5,
                        y0=y_idx - 0.5,
                        y1=y_idx + 0.5,
                        line=dict(color="black", width=1),
                        fillcolor="rgba(0,0,0,0)",
                        layer="above",
                    )

        # Save and display plot as interactive HTML page
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"outputs/plots/probs_{timestamp}.html"
        fig.write_html(filename)

        fig.show(renderer="browser")

    else:
        fig, ax = plt.subplots()

        fig.set_figheight(5)
        fig.set_figwidth(7)

        ax.imshow(
            data,
            cmap="YlOrBr",
            origin="lower",
        )

        ax.grid(
            which="minor",
            axis="both",
            linestyle="-",
            color="k",
            linewidth=0.5,
        )

        n_x_vals = len(x_labels)
        ax.set_xticks(np.arange(0, n_x_vals))
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_xlabel("Position in generated sequence")

        n_y_vals = len(y_labels)
        ax.set_yticks(np.arange(0, n_y_vals))
        ax.set_yticklabels(y_labels)
        ax.set_ylabel("Top 20 candidate output tokens (indexed by probability)")

        for y_idx in range(len(grid_labels)):
            for x_idx in range(len(grid_labels[0])):
                ax.annotate(
                    grid_labels[y_idx][x_idx], (x_idx, y_idx), fontsize=3, rotation=45
                )

        cbar = plt.colorbar(ax.images[0], ax=ax, shrink=0.75)
        cbar.set_label("Log probability of token")

        # Save plot as static PNG image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"outputs/plots/probs_{timestamp}.png"
        plt.savefig(filename)


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
    user_prompt = 'Summarise the following news article in 30 words. Article: \n A report revealed that 253 potential victims of slavery were reported in Hampshire and the Isle of Wight, of which one in four were children. Modern slavery, which includes human trafficking, is the illegal exploitation of people for personal or commercial gain. It can take different forms of slavery, such as domestic or labour exploitation, organ harvesting, EU Status exploitation, and financial, sexual and criminal exploitation. Each year, Hampshire and Isle of Wight Fire and Rescue Authority (HIWFRA), combined by all four authorities, the three unitary councils, and the county council, spends around £99m on making "life safer" in the county and preventing slavery and human trafficking. However, a recent report of the HIWFRA has revealed that by June 2023, there were 253 potential victims identified of modern slavery in Hampshire and the Isle of Wight. Of them, one in four were children. According to the Government\'s UK Annual Report on Modern Slavery, 10,613 potential victims were referred to the National Referral Mechanism in the year ended September 2021. In case any member of the Authority or any of its staff suspects slavery or human trafficking activity either within the community or the organisation, then the concerns will be reported through the Service\'s Safeguarding Reporting Procedure. If slavery or human trafficking activity is suspected through its supply chain, it will be reported to Hampshire Constabulary via the Modern Slavery Helpline. If you need help, advice or information about any modern slavery issue, you can contact the modern slavery helpline confidentially, 24 hours a day, 365 days a year on 0800 012 1700"'  # noqa
    output_distribution_visualiser(
        user_prompt=user_prompt, save_model_distribution=True
    )

    # pregenerated_tokens_file = "outputs/logprobs/test_20251002.json"
    # output_distribution_visualiser(restore_from_file=pregenerated_tokens_file)
