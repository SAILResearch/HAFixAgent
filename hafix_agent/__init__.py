"""
HAFixAgent: History-Aware Automated Program Repair Agent

A dataset-agnostic automated program repair agent that extends mini-swe-agent 
with history-aware blame context extraction.
"""

__version__ = "0.1.0"
__author__ = "HAFixAgent"

import sys
from rich.console import Console

def show_hafix_greeting():
    """Show HAFixAgent greeting with attribution to mini-swe-agent."""
    console = Console(stderr=True)
    console.print("🌿 HAFixAgent (history-aware automated program repair agent, built on mini-swe-agent)")

# Show HAFixAgent greeting when imported in a CLI context
if hasattr(sys, 'argv') and sys.argv:
    show_hafix_greeting()