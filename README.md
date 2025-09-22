#CtrlNews: LLM-based Multi-Agent Controllable News Writing via Knowledge Gravitational Field has been accepted to EMNLP 2025

## Overview

This is an automated news generation system powered by large language models (LLMs) and a "gravitational field" approach. The system automatically generates news articles with controllable sentiment through multi-agent discussions, knowledge graph expansion, and opinion generation.

## Key Features

- Multi-agent discussion to generate initial question sets
- Directed gravitational field model construction
- Automated web search for relevant information
- Multi-round knowledge graph expansion
- Sentiment-controlled opinion generation (adjustable positive/neutral/negative ratios)
- Automated article writing and optimization
- Batch generation of multi-topic news articles

## Technical Architecture

- Based on DeepSeek series large language models
- DuckDuckGo search tool for information retrieval
- NewsGravityField for knowledge structure representation
- ArticleNewsWriter for article generation
- SimpleArticleReviewer for article evaluation and optimization

## Installation

### Prerequisites

- Python 3.8+
- Anaconda environment (recommended)

### Installation Steps

1. Clone the repository
2. Create and activate a new conda environment
   ```bash
   conda create -n ctrlnews python=3.8
   conda activate ctrlnews
   ```
3. Install the dependencies

## Usage

### Configuration

In the `main` function, you can modify the following configuration parameters:

- `topics_file`: Path to the topics JSON file
- `output_dir`: Output directory for generated articles
- `generations_per_topic`: Number of articles to generate per topic
- `sentiment_ratios`: Sentiment ratio configuration
- `max_expansion_rounds`: Maximum knowledge expansion rounds

### Running

```bash
python generate_controlled_wf_gra_full.py
```

## Project Structure

```
./
├── data/                     # Data directory
│   ├── news_topics.json      # News topics configuration
│   └── workflow/             # Generated news articles
├── tools/                    # Tool classes
│   ├── duckduckgo_searchtool.py  # Search tool
│   └── lm.py                 # Language model tools
├── GravitionalField.py       # Gravitational field model
├── ArticleNewsWriter.py      # Article generator
├── simple_reviewer.py        # Article review component
└── generate_controlled_wf_gra_full.py  # Main program
```

## Workflow

1. Load topics from configuration file
2. For each topic:
   - Conduct multi-agent discussion to generate initial questions
   - Build directed gravitational field and add questions
   - Search and answer questions using DuckDuckGo
   - Build gravitational connections
   - Expand knowledge graph through multiple rounds
   - Apply sentiment control to peripheral nodes
   - Generate opinions for peripheral nodes
   - Generate news article using the opinions
   - Review and optimize the article
   - Save the article and used opinions

## Notes

- Ensure sufficient API access rights and quotas
- Generating numerous articles may require significant time
- Stable network connection is required for search functionality
- The sentiment ratio can be adjusted to control the opinion bias

## Future Improvements

- Add more language models support
- Improve search capabilities
- Enhance article generation quality
- Add more topics and domains

## License

This project is licensed under the MIT License - see the LICENSE file for details.
