import matplotlib.pyplot as plt
from pathlib import Path
import re
import spacy
import networkx as nx
from wordcloud import WordCloud
from collections import Counter, defaultdict
import plotly.graph_objs as go
from tools import DATA_DIR

# Load the spaCy model
nlp = spacy.load('en_core_web_sm', disable=["parser"])  # Disable the parser to save memory
nlp.add_pipe('sentencizer')

def preprocess_text(text):
    """Tokenizes text, removes stopwords and punctuation."""
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return tokens

def find_cooccurrences(tokens, n=2):
    """Finds n-grams in tokens and counts their frequencies."""
    n_grams = zip(*[tokens[i:] for i in range(n)])
    n_gram_freq = Counter(n_grams)
    return n_gram_freq


def plot_cooccurrence_network_plotly(n_gram_freq, top_n=15, output_png='cooccurrence_network.png'):
    """Plots a co-occurrence network using Plotly."""
    G = nx.Graph()

    # Add edges for the most common co-occurrences
    for (word1, word2), freq in n_gram_freq.most_common(top_n):
        G.add_edge(word1, word2, weight=freq)

    # Assign positions for each node in 2D space
    pos = nx.spring_layout(G, seed=42, k=1.5)

    # Create edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(color='gray', width=0.3 * G.edges[edge]['weight']),
            hoverinfo='none'
        ))

    # Create node trace without text labels
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers',
        marker=dict(
            size=60,
            color='lightblue',
            line=dict(width=2, color='black')
        ),
        hoverinfo='text'
    )

    # Create annotations for node labels
    annotations = [
        dict(
            x=pos[node][0],
            y=pos[node][1],
            xref='x',
            yref='y',
            text=node,
            showarrow=False,
            font=dict(size=20, color='black'),
            align='center',
            xshift=10,
            yshift=10
        ) for node in G.nodes()
    ]

    # Adjust the plot height
    plot_height = 700

    # Create the layout
    layout = go.Layout(
        title='Co-occurrence Network of Word Pairs',
        showlegend=False,
        margin=dict(b=40, l=40, r=40, t=40),
        width=1000,
        height=plot_height,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        annotations=annotations
    )

    # Plot the figure
    fig = go.Figure(data=edge_trace + [node_trace], layout=layout)

    # Save the figure as PNG
    fig.write_image(DATA_DIR / output_png)
    return output_png


def cooccurrence_analysis(text):
    """Performs co-occurrence analysis and generates a network plot."""
    tokens = preprocess_text(text)
    n_gram_freq = find_cooccurrences(tokens, n=2)
    path = plot_cooccurrence_network_plotly(n_gram_freq)
    return path


def analyze_speaker_turns(transcript):
    """Analyzes word and sentence counts for each speaker."""
    speaker_word_counts = defaultdict(int)
    speaker_sentence_counts = defaultdict(int)

    lines = transcript.strip().split('\n')
    speaker_pattern = re.compile(r'^(Speaker [A-Z][a-zA-Z0-9]*):\s*(.*)$')

    for line in lines:
        match = speaker_pattern.match(line)
        if match:
            speaker = match.group(1)
            dialogue = match.group(2)

            # Count words
            word_count = len(dialogue.split())
            speaker_word_counts[speaker] += word_count

            # Count sentences using spaCy
            doc = nlp(dialogue)
            sentence_count = len(list(doc.sents))
            speaker_sentence_counts[speaker] += sentence_count

    return speaker_word_counts, speaker_sentence_counts


def plot_speaker_turns(speaker_word_counts, speaker_sentence_counts):
    """Plots word and sentence counts for each speaker."""
    # Ensure that both dictionaries have the same speakers
    speakers = list(speaker_word_counts.keys())
    word_counts = [speaker_word_counts[speaker] for speaker in speakers]
    sentence_counts = [speaker_sentence_counts.get(speaker, 0) for speaker in speakers]

    # Check if there are speakers to plot
    if not speakers:
        print("No speaker data available to plot.")
        return None

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Plot word counts
    axes[0].bar(speakers, word_counts, color='lightblue')
    axes[0].set_title('Number of Words Spoken by Each Speaker')
    axes[0].set_xlabel('Speakers')
    axes[0].set_ylabel('Word Count')

    # Plot sentence counts
    axes[1].bar(speakers, sentence_counts, color='lightgreen')
    axes[1].set_title('Number of Sentences Spoken by Each Speaker')
    axes[1].set_xlabel('Speakers')
    axes[1].set_ylabel('Sentence Count')

    plt.tight_layout()

    # Ensure DATA_DIR exists
    output_dir = DATA_DIR if DATA_DIR.exists() else Path('.')
    file_name = 'speaker_turns.png'
    output_path = output_dir / file_name

    plt.savefig(output_path)
    plt.close()

    return str(output_path)


def generate_word_cloud(text):
    """Generates a word cloud from the text."""
    filtered_words = preprocess_text(text)
    word_counts = Counter(filtered_words)

    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap='viridis').generate_from_frequencies(word_counts)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    file_name = 'word_cloud.png'
    plt.savefig(DATA_DIR / file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

    return file_name

def named_entity_analysis(text):
    """Performs NER analysis and plots the most common entities."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    entity_counts = Counter(entities)

    # Get the most common entities
    most_common_entities = entity_counts.most_common(10)

    # Plot the entity counts
    entities = [entity for entity, count in most_common_entities]
    counts = [count for entity, count in most_common_entities]

    plt.figure(figsize=(10, 6))
    plt.bar(entities, counts, color='orange')
    plt.title('Top Named Entities')
    plt.xlabel('Entities')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()

    file_name = 'named_entities.png'
    plt.savefig(DATA_DIR / file_name)
    plt.close()

    return file_name

def analyze_transcript(transcript):
    """Runs all analyses on the transcript and returns paths to generated images."""
    speaker_word_counts, speaker_sentence_counts = analyze_speaker_turns(transcript)
    return {
        "Speaker Turns": plot_speaker_turns(speaker_word_counts, speaker_sentence_counts),
        "Word Cloud": generate_word_cloud(transcript),
        "Co-occurrence Analysis": cooccurrence_analysis(transcript),
        "Named Entity Analysis": named_entity_analysis(transcript)
    }


if __name__ == '__main__':
    # Sample transcript from text file
    with open('AB1_Discussion_transcript.txt', 'r') as file:
        transcript = file.read()
    # Analyze the transcript
    analyze_transcript(transcript)
