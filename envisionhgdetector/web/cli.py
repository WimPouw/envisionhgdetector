#!/usr/bin/env python
# envisionhgdetector/web/cli.py
"""
Command-line interface for launching the EnvisionHG web interface.
"""

import argparse
import webbrowser
import time
import threading


def main():
    """Main entry point for the web CLI."""
    parser = argparse.ArgumentParser(
        description='EnvisionHG Web Interface - Gesture Detection Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  envisionhg-web                    # Start on default port 5000
  envisionhg-web --port 8080        # Start on port 8080
  envisionhg-web --no-browser       # Don't open browser automatically
  envisionhg-web --debug            # Enable debug mode
        '''
    )

    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Port to run on (default: 5000)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Don\'t open browser automatically'
    )

    args = parser.parse_args()

    # Import here to avoid loading heavy dependencies unless needed
    from .app import run_server

    # Open browser after a short delay
    if not args.no_browser:
        url = f'http://{args.host}:{args.port}'

        def open_browser():
            time.sleep(1.5)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

    # Run server
    run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
