#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Upgrade or install gpt4all package
pip install --upgrade --quiet gpt4all

# Shell completion setup

# Detect the shell
shell=$(basename "$SHELL")

echo "Setting up shell completion for your $shell shell."

# Generate the completion script
completion_script=$(python chat.py --show-completion)

if [[ "$shell" == "bash" ]]; then
    # Install for Bash
    mkdir -p ~/.bash_completion.d
    echo "$completion_script" > ~/.bash_completion.d/chat_completion
    # Ensure completion is sourced
    if ! grep -q 'source ~/.bash_completion.d/chat_completion' ~/.bashrc; then
        echo "source ~/.bash_completion.d/chat_completion" >> ~/.bashrc
    fi
    echo "Bash completion installed! Please restart your terminal or run 'source ~/.bashrc' to activate."

elif [[ "$shell" == "zsh" ]]; then
    # Install for Zsh
    mkdir -p ~/.zsh/completions
    echo "$completion_script" > ~/.zsh/completions/_chat
    echo "Zsh completion installed! Please restart your terminal or run 'source ~/.zshrc' to activate."

elif [[ "$shell" == "fish" ]]; then
    # Install for Fish
    mkdir -p ~/.config/fish/completions
    echo "$completion_script" > ~/.config/fish/completions/chat.fish
    echo "Fish completion installed! Please restart your terminal or run 'source ~/.config/fish/completions/chat.fish' to activate."

else
    echo "Unsupported shell: $shell"
fi
