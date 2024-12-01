# infrahub_react_agent
An artificial intelligence ReAct Agent for Infrahub by OpsMill 

Welcome to the Infrahub AI Agent project! This application provides a natural language interface for interacting with Infrahub APIs, enabling CRUD (Create, Read, Update, Delete) operations through an intuitive chat-like interface powered by AI.

This project simplifies network management by combining AI-driven agents with the Infrahub API to streamline and automate common network tasks.

## Disclamer

This is a fork from [https://github.com/automateyournetwork/netbox_react_agent](https://github.com/automateyournetwork/netbox_react_agent) by John Capobianco. The original project was focusing on NetBox and this is an early draft to adapt it for Infrahub. Do not use it in production. 

## Branches Overview

### Main Branch

Powered by ChatGPT (gpt-4o)

**Requires OpenAI API Key**

Offers high accuracy and performance for handling natural language queries.

Recommended for production use.

API costs apply.

## Features

Natural Language Interface: Interact with Infrahub APIs using plain English commands.

CRUD Operations: Perform Create, Read, Update, and Delete tasks on your Infrahub data.

*API Validation: Ensures commands align with supported Infrahub API endpoints.*

Dynamic Tools: Auto-detects and leverages the appropriate tools for each task.

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed.
- OpenAI API Key

## Quick Start

### Clone the Repository

``` bash
git clone https://github.com/ubaumann/infrahub_react_agent.git
cd infrahub_ai_agent
```

### Run the Application

```bash
docker compose up
```

### Access the App

Open your browser and go to http://localhost:8501.

Configure API Keys

## For the main branch:

Provide your Infrahub Base URL, Infrahub Token, and OpenAI API Key in the configuration page.

# Start Chatting

Use natural language to manage your Infrahub data. Example commands:

"Fetch all devices in the DC1 site."

"Create a new VLAN in site DC2."

## Key Components

InfrahubController: Manages interactions with the Infrahub API.

LangChain ReAct Agent: Dynamically selects tools to process natural language queries.

Streamlit Interface: Provides an intuitive chat-like web UI.

## Troubleshooting

Docker Issues: Ensure Docker is running and your system meets the necessary prerequisites.

OpenAI Key Errors: Check that your API key is valid and added correctly.

Infrahub API Errors: Verify your Infrahub instance is accessible, and the API token has the required permissions.
