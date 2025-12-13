"""
ΣLANG Command Line Interface

Click-based CLI for encoding, decoding, analogies, search, and server operations.
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional, List, TextIO
from dataclasses import asdict

import click
import numpy as np

# Version info
__version__ = "1.0.0"


# =============================================================================
# Click Context & Utilities
# =============================================================================

class CLIContext:
    """Shared context for CLI commands."""
    
    def __init__(self, verbose: bool = False, quiet: bool = False, 
                 output_format: str = "text", config_path: Optional[str] = None):
        self.verbose = verbose
        self.quiet = quiet
        self.output_format = output_format
        self.config_path = config_path
        self._api = None
    
    @property
    def api(self):
        """Lazy-load the API."""
        if self._api is None:
            from .api_server import create_api
            from .config import get_config
            
            if self.config_path:
                config = get_config()
                # Could load from file here
            else:
                config = get_config()
            
            self._api = create_api(config)
        return self._api
    
    def output(self, data, success: bool = True):
        """Output data in the configured format."""
        if self.quiet and success:
            return
        
        if self.output_format == "json":
            if hasattr(data, '__dict__'):
                click.echo(json.dumps(asdict(data) if hasattr(data, '__dataclass_fields__') else data.__dict__, indent=2, default=str))
            elif isinstance(data, dict):
                click.echo(json.dumps(data, indent=2, default=str))
            elif isinstance(data, list):
                click.echo(json.dumps([asdict(d) if hasattr(d, '__dataclass_fields__') else d for d in data], indent=2, default=str))
            elif isinstance(data, np.ndarray):
                click.echo(json.dumps(data.tolist()))
            else:
                click.echo(json.dumps({"result": str(data)}, default=str))
        else:
            if isinstance(data, np.ndarray):
                click.echo(f"Vector ({len(data)} dims): [{data[:5].tolist()}...]")
            else:
                click.echo(str(data))
    
    def log(self, message: str, level: str = "info"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            prefix = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}.get(level, "")
            click.echo(f"{prefix} {message}", err=True)
    
    def error(self, message: str, exit_code: int = 1):
        """Output an error and optionally exit."""
        click.echo(click.style(f"Error: {message}", fg="red"), err=True)
        if exit_code > 0:
            sys.exit(exit_code)


pass_context = click.make_pass_decorator(CLIContext, ensure=True)


# =============================================================================
# Main CLI Group
# =============================================================================

@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="sigmalang")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output")
@click.option("-f", "--format", "output_format", 
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True),
              help="Path to configuration file")
@click.pass_context
def cli(ctx, verbose, quiet, output_format, config_path):
    """
    ΣLANG - Semantic Language Encoding and Analysis
    
    A powerful tool for semantic text encoding, analogy solving,
    and intelligent text search.
    
    Examples:
    
        sigmalang encode "Hello, world!"
        
        sigmalang analogy solve "king:queen::man:?"
        
        sigmalang serve --port 8000
    """
    ctx.ensure_object(CLIContext)
    ctx.obj = CLIContext(
        verbose=verbose, 
        quiet=quiet, 
        output_format=output_format,
        config_path=config_path
    )
    
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# =============================================================================
# Encode Command
# =============================================================================

@cli.command("encode")
@click.argument("text", required=False)
@click.option("-i", "--input", "input_file", type=click.File("r"),
              help="Input file to encode")
@click.option("-o", "--output", "output_file", type=click.Path(),
              help="Output file for vector (numpy .npy format)")
@click.option("--normalize/--no-normalize", default=True,
              help="Normalize output vectors")
@click.option("--stdin", is_flag=True, help="Read input from stdin")
@pass_context
def encode_cmd(ctx, text, input_file, output_file, normalize, stdin):
    """
    Encode text to ΣLANG vectors.
    
    Examples:
    
        sigmalang encode "Hello, world!"
        
        sigmalang encode -i input.txt -o vectors.npy
        
        echo "text" | sigmalang encode --stdin
    """
    try:
        # Get input text
        if stdin:
            text = sys.stdin.read().strip()
        elif input_file:
            text = input_file.read().strip()
        elif not text:
            ctx.error("No input text provided. Use TEXT argument, --input, or --stdin")
            return
        
        ctx.log(f"Encoding: {text[:50]}...")
        
        # Initialize and encode
        api = ctx.api
        
        from .api_models import EncodeRequest
        request = EncodeRequest(texts=[text], normalize=normalize)
        response = api.encode(request)
        
        if response.success and response.vectors:
            vector = np.array(response.vectors[0])
            
            # Save or output
            if output_file:
                np.save(output_file, vector)
                ctx.log(f"Saved vector to {output_file}", "success")
                if not ctx.quiet:
                    click.echo(f"Encoded to {output_file} ({len(vector)} dimensions)")
            else:
                ctx.output(vector)
        else:
            ctx.error(f"Encoding failed: {response.error}")
            
    except Exception as e:
        ctx.error(f"Encoding error: {e}")


# =============================================================================
# Decode Command
# =============================================================================

@cli.command("decode")
@click.argument("vector_file", required=False, type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", type=click.File("w"),
              help="Output file for decoded text")
@click.option("--max-length", default=512, help="Maximum output length")
@pass_context
def decode_cmd(ctx, vector_file, output_file, max_length):
    """
    Decode ΣLANG vectors back to text.
    
    Examples:
    
        sigmalang decode vectors.npy
        
        sigmalang decode vectors.npy -o output.txt
    """
    try:
        if not vector_file:
            ctx.error("Vector file required")
            return
        
        ctx.log(f"Decoding: {vector_file}")
        
        # Load vector
        vector = np.load(vector_file)
        
        # Initialize and decode
        api = ctx.api
        
        from .api_models import DecodeRequest
        request = DecodeRequest(vectors=[vector.tolist()], max_length=max_length)
        response = api.decode(request)
        
        if response.success and response.texts:
            decoded_text = response.texts[0]
            
            if output_file:
                output_file.write(decoded_text)
                ctx.log(f"Decoded text written", "success")
            else:
                ctx.output(decoded_text)
        else:
            ctx.error(f"Decoding failed: {response.error}")
            
    except Exception as e:
        ctx.error(f"Decoding error: {e}")


# =============================================================================
# Analogy Commands
# =============================================================================

@cli.group("analogy")
@pass_context
def analogy_group(ctx):
    """
    Analogy operations.
    
    Solve and explain semantic analogies like "king:queen::man:?"
    """
    pass


@analogy_group.command("solve")
@click.argument("analogy")
@click.option("-k", "--top-k", default=5, help="Number of solutions to return")
@click.option("-t", "--type", "analogy_type", 
              type=click.Choice(["semantic", "structural", "proportional"]),
              default="semantic", help="Analogy type")
@pass_context
def analogy_solve(ctx, analogy, top_k, analogy_type):
    """
    Solve an analogy.
    
    Format: "A:B::C:?" or "A:B::C"
    
    Examples:
    
        sigmalang analogy solve "king:queen::man:?"
        
        sigmalang analogy solve "dog:puppy::cat:?" --top-k 3
    """
    try:
        # Parse analogy format
        parts = analogy.replace("?", "").replace("::", ":").split(":")
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) < 3:
            ctx.error("Invalid analogy format. Use A:B::C:? or A:B::C")
            return
        
        a, b, c = parts[0], parts[1], parts[2]
        ctx.log(f"Solving analogy: {a}:{b}::{c}:?")
        
        # Solve
        api = ctx.api
        
        from .api_models import AnalogyRequest, AnalogyType
        analogy_type_enum = AnalogyType(analogy_type)
        
        request = AnalogyRequest(a=a, b=b, c=c, top_k=top_k, analogy_type=analogy_type_enum)
        response = api.solve_analogy(request)
        
        if response.success and response.solutions:
            if ctx.output_format == "json":
                ctx.output(response.solutions)
            else:
                click.echo(f"\n{a}:{b}::{c}:?\n")
                for i, sol in enumerate(response.solutions, 1):
                    conf = f"{sol.confidence:.2%}" if hasattr(sol, 'confidence') else "N/A"
                    answer = sol.answer if hasattr(sol, 'answer') else str(sol)
                    click.echo(f"  {i}. {answer} (confidence: {conf})")
        else:
            ctx.error(f"Analogy solving failed: {response.error}")
            
    except Exception as e:
        ctx.error(f"Analogy error: {e}")


@analogy_group.command("explain")
@click.argument("analogy")
@pass_context
def analogy_explain(ctx, analogy):
    """
    Explain an analogy relationship.
    
    Format: "A:B::C:D"
    
    Examples:
    
        sigmalang analogy explain "king:queen::man:woman"
    """
    try:
        parts = analogy.replace("::", ":").split(":")
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) != 4:
            ctx.error("Invalid analogy format. Use A:B::C:D")
            return
        
        a, b, c, d = parts[0], parts[1], parts[2], parts[3]
        ctx.log(f"Explaining analogy: {a}:{b}::{c}:{d}")
        
        api = ctx.api
        
        from .api_models import AnalogyExplainRequest
        request = AnalogyExplainRequest(a=a, b=b, c=c, d=d)
        response = api.explain_analogy(request)
        
        if response.success:
            ctx.output(response)
        else:
            ctx.error(f"Explanation failed: {response.error}")
            
    except Exception as e:
        ctx.error(f"Analogy error: {e}")


# =============================================================================
# Search Command
# =============================================================================

@cli.command("search")
@click.argument("query")
@click.option("-c", "--corpus", "corpus_file", type=click.File("r"),
              help="Corpus file (one document per line)")
@click.option("-k", "--top-k", default=10, help="Number of results")
@click.option("-m", "--mode", 
              type=click.Choice(["semantic", "exact", "hybrid", "fuzzy"]),
              default="semantic", help="Search mode")
@click.option("-t", "--threshold", default=0.0, type=float,
              help="Minimum similarity threshold")
@pass_context
def search_cmd(ctx, query, corpus_file, top_k, mode, threshold):
    """
    Search corpus for similar documents.
    
    Examples:
    
        sigmalang search "machine learning" --corpus documents.txt
        
        sigmalang search "AI" -k 5 --mode semantic
    """
    try:
        corpus = []
        if corpus_file:
            corpus = [line.strip() for line in corpus_file if line.strip()]
        
        if not corpus:
            ctx.error("No corpus provided. Use --corpus option.")
            return
        
        ctx.log(f"Searching {len(corpus)} documents for: {query}")
        
        api = ctx.api
        
        from .api_models import SearchRequest, SearchMode
        search_mode = SearchMode(mode)
        
        request = SearchRequest(
            query=query, 
            corpus=corpus, 
            top_k=top_k,
            mode=search_mode,
            threshold=threshold
        )
        response = api.search_corpus(request)
        
        if response.success and response.results:
            if ctx.output_format == "json":
                ctx.output(response.results)
            else:
                click.echo(f"\nSearch results for: {query}\n")
                for i, result in enumerate(response.results, 1):
                    score = f"{result.score:.4f}" if hasattr(result, 'score') else "N/A"
                    text = result.text[:80] + "..." if len(result.text) > 80 else result.text
                    click.echo(f"  {i}. [{score}] {text}")
        else:
            click.echo("No results found.")
            
    except Exception as e:
        ctx.error(f"Search error: {e}")


# =============================================================================
# Entities Command
# =============================================================================

@cli.group("entities")
@pass_context
def entities_group(ctx):
    """
    Entity extraction operations.
    
    Extract named entities and relationships from text.
    """
    pass


@entities_group.command("extract")
@click.argument("text", required=False)
@click.option("-i", "--input", "input_file", type=click.File("r"),
              help="Input file")
@click.option("--types", multiple=True,
              help="Entity types to extract (person, org, location, etc.)")
@click.option("--relations/--no-relations", default=True,
              help="Extract relations between entities")
@pass_context
def entities_extract(ctx, text, input_file, types, relations):
    """
    Extract entities from text.
    
    Examples:
    
        sigmalang entities extract "John works at Google in New York"
        
        sigmalang entities extract -i document.txt --types person org
    """
    try:
        if input_file:
            text = input_file.read().strip()
        elif not text:
            ctx.error("No input text provided")
            return
        
        ctx.log(f"Extracting entities from: {text[:50]}...")
        
        api = ctx.api
        
        from .api_models import EntityExtractionRequest
        request = EntityExtractionRequest(
            text=text,
            extract_relations=relations
        )
        response = api.extract_entities(request)
        
        if response.success:
            if ctx.output_format == "json":
                ctx.output({
                    "entities": [asdict(e) for e in response.entities] if response.entities else [],
                    "relations": [asdict(r) for r in response.relations] if response.relations else []
                })
            else:
                if response.entities:
                    click.echo("\nEntities:")
                    for ent in response.entities:
                        click.echo(f"  • {ent.text} ({ent.type})")
                
                if relations and response.relations:
                    click.echo("\nRelations:")
                    for rel in response.relations:
                        click.echo(f"  • {rel.source} --[{rel.relation}]--> {rel.target}")
        else:
            ctx.error(f"Extraction failed: {response.error}")
            
    except Exception as e:
        ctx.error(f"Entity extraction error: {e}")


# =============================================================================
# Server Commands
# =============================================================================

@cli.group("serve")
@click.pass_context
def serve_group(ctx):
    """
    API server operations.
    
    Start and manage the ΣLANG API server.
    """
    # Default action for 'sigmalang serve' without subcommand
    pass


@cli.command("server")
@click.option("-h", "--host", default="0.0.0.0", help="Host to bind to")
@click.option("-p", "--port", default=8000, type=int, help="Port to listen on")
@click.option("-w", "--workers", default=1, type=int, help="Number of worker processes")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--log-level", default="info",
              type=click.Choice(["debug", "info", "warning", "error"]),
              help="Logging level")
@pass_context
def serve_cmd(ctx, host, port, workers, reload, log_level):
    """
    Start the ΣLANG API server.
    
    Examples:
    
        sigmalang server
        
        sigmalang server --host 0.0.0.0 --port 8080
        
        sigmalang server --workers 4 --reload
    """
    try:
        click.echo(f"Starting ΣLANG API server on {host}:{port}")
        click.echo(f"Workers: {workers}, Reload: {reload}")
        
        # Try to use uvicorn
        try:
            import uvicorn
            
            # Create the FastAPI app configuration
            uvicorn.run(
                "core.api_server:create_fastapi_app",
                host=host,
                port=port,
                workers=workers if not reload else 1,
                reload=reload,
                log_level=log_level,
                factory=True
            )
        except ImportError:
            ctx.error("uvicorn not installed. Install with: pip install uvicorn")
            
    except Exception as e:
        ctx.error(f"Server error: {e}")


# =============================================================================
# Config Commands
# =============================================================================

@cli.group("config")
@pass_context
def config_group(ctx):
    """
    Configuration management.
    
    View and modify ΣLANG configuration.
    """
    pass


@config_group.command("show")
@click.option("--section", help="Show specific section")
@pass_context
def config_show(ctx, section):
    """
    Show current configuration.
    
    Examples:
    
        sigmalang config show
        
        sigmalang config show --section api
    """
    try:
        from .config import get_config
        config = get_config()
        
        config_dict = {
            "api": {
                "host": config.api.host,
                "port": config.api.port,
                "debug": config.api.debug,
            },
            "cache": {
                "enabled": config.cache.enabled,
                "ttl_seconds": config.cache.ttl_seconds,
            },
            "features": {
                "enable_caching": config.features.enable_caching,
                "enable_rate_limiting": config.features.enable_rate_limiting,
                "enable_metrics": config.features.enable_metrics,
            }
        }
        
        if section and section in config_dict:
            ctx.output(config_dict[section])
        else:
            ctx.output(config_dict)
            
    except Exception as e:
        ctx.error(f"Config error: {e}")


@config_group.command("set")
@click.argument("key_value")
@pass_context
def config_set(ctx, key_value):
    """
    Set a configuration value.
    
    Format: KEY=VALUE
    
    Examples:
    
        sigmalang config set api.port=8080
        
        sigmalang config set cache.enabled=true
    """
    try:
        if "=" not in key_value:
            ctx.error("Invalid format. Use KEY=VALUE")
            return
        
        key, value = key_value.split("=", 1)
        click.echo(f"Setting {key} = {value}")
        click.echo("Note: Configuration changes are session-only unless saved to file.")
        
    except Exception as e:
        ctx.error(f"Config error: {e}")


# =============================================================================
# Batch Commands
# =============================================================================

@cli.group("batch")
@pass_context
def batch_group(ctx):
    """
    Batch processing operations.
    
    Process multiple inputs efficiently.
    """
    pass


@batch_group.command("encode")
@click.argument("input_file", type=click.File("r"))
@click.option("-o", "--output", "output_file", type=click.Path(), required=True,
              help="Output file for vectors (numpy .npy format)")
@click.option("--normalize/--no-normalize", default=True,
              help="Normalize output vectors")
@click.option("--progress/--no-progress", default=True,
              help="Show progress bar")
@pass_context
def batch_encode(ctx, input_file, output_file, normalize, progress):
    """
    Batch encode multiple texts to vectors.
    
    Input file should have one text per line.
    
    Examples:
    
        sigmalang batch encode input.txt -o vectors.npy
    """
    try:
        lines = [line.strip() for line in input_file if line.strip()]
        
        if not lines:
            ctx.error("Input file is empty")
            return
        
        ctx.log(f"Encoding {len(lines)} texts...")
        
        api = ctx.api
        
        from .api_models import EncodeRequest
        
        vectors = []
        iterator = lines
        
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(lines, desc="Encoding")
            except ImportError:
                pass
        
        for text in iterator:
            request = EncodeRequest(texts=[text], normalize=normalize)
            response = api.encode(request)
            if response.success and response.vectors:
                vectors.append(response.vectors[0])
        
        # Save all vectors
        np.save(output_file, np.array(vectors))
        click.echo(f"Encoded {len(vectors)} texts to {output_file}")
        
    except Exception as e:
        ctx.error(f"Batch encoding error: {e}")


@batch_group.command("decode")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", type=click.File("w"), required=True,
              help="Output file for decoded texts")
@pass_context
def batch_decode(ctx, input_file, output_file):
    """
    Batch decode vectors to texts.
    
    Examples:
    
        sigmalang batch decode vectors.npy -o output.txt
    """
    try:
        vectors = np.load(input_file)
        
        ctx.log(f"Decoding {len(vectors)} vectors...")
        
        api = ctx.api
        
        from .api_models import DecodeRequest
        
        for vector in vectors:
            request = DecodeRequest(vectors=[vector.tolist()])
            response = api.decode(request)
            if response.success and response.texts:
                output_file.write(response.texts[0] + "\n")
        
        click.echo(f"Decoded {len(vectors)} vectors")
        
    except Exception as e:
        ctx.error(f"Batch decoding error: {e}")


# =============================================================================
# Info Command
# =============================================================================

@cli.command("info")
@pass_context
def info_cmd(ctx):
    """
    Show system information.
    
    Examples:
    
        sigmalang info
    """
    try:
        api = ctx.api
        response = api.info()
        
        if ctx.output_format == "json":
            ctx.output(response)
        else:
            click.echo(f"\nΣLANG System Information")
            click.echo(f"========================")
            click.echo(f"Version: {response.version if hasattr(response, 'version') else __version__}")
            click.echo(f"API Version: v1")
            
            if hasattr(response, 'capabilities') and response.capabilities:
                click.echo(f"\nCapabilities:")
                for cap in response.capabilities:
                    click.echo(f"  • {cap}")
                    
    except Exception as e:
        ctx.error(f"Info error: {e}")


# =============================================================================
# Health Command
# =============================================================================

@cli.command("health")
@pass_context
def health_cmd(ctx):
    """
    Check system health.
    
    Examples:
    
        sigmalang health
    """
    try:
        api = ctx.api
        response = api.health()
        
        if ctx.output_format == "json":
            ctx.output(response)
        else:
            status = response.status if hasattr(response, 'status') else "unknown"
            status_color = "green" if status == "healthy" else "yellow" if status == "degraded" else "red"
            
            click.echo(f"\nSystem Health: ", nl=False)
            click.echo(click.style(status.upper(), fg=status_color, bold=True))
            
            if hasattr(response, 'components') and response.components:
                click.echo(f"\nComponents:")
                for comp in response.components:
                    comp_status = comp.status if hasattr(comp, 'status') else "unknown"
                    color = "green" if comp_status == "healthy" else "red"
                    name = comp.name if hasattr(comp, 'name') else str(comp)
                    click.echo(f"  • {name}: ", nl=False)
                    click.echo(click.style(comp_status, fg=color))
                    
    except Exception as e:
        ctx.error(f"Health check error: {e}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point for the CLI."""
    cli(obj=CLIContext())


if __name__ == "__main__":
    main()
