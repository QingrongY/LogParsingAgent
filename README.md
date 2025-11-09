# Agent-based Log Parser

Semantic-aware log parsing using LLM agents to distinguish structural elements from business data.

## Core Concept

Traditional log parsers struggle to distinguish between:
- **Branches (structural)**: Event types, log levels, protocol keywords - finite, system-defined value sets
- **Fruits (business data)**: IPs, timestamps, user IDs - unbounded, instance-specific values

This system uses LLM agents to make this distinction semantically, not heuristically.

## Architecture

### Multi-Agent System
- **RouterAgent**: Identifies log source (device type, vendor)
- **TimestampAgent**: Infers timestamp format
- **ParsingAgent**: Derives templates using LLM semantic analysis
- **JSONPayloadPreprocessor**: Detects inline JSON-like payloads, replaces them with placeholders, and captures structured key metadata for downstream processing
- **TemplateValidator**: Performs conflict checks before new templates enter the library
- **ConflictResolutionAgent**: Resolves template conflicts
- **TemplateRefinementAgent**: Refines templates to meet validation requirements

### Key Components
- **TemplateLibrary**: Persistent storage of learned templates per device/vendor

## Installation

```bash
pip install -r requirements.txt
```

Create `config.json` with your API key:
```json
{
  "AIML_API_KEY": "your-api-key-here"
}
```

## Usage

```bash
# Basic usage
python cli.py /path/to/logfile.log

# Specify output directory
python cli.py /path/to/logfile.log --output-dir ./results
```

## Output

- `*.parsed.json`: Structured parsing results with extracted variables
- Each record includes a `json_payloads` array detailing placeholder-bound payloads, extracted keys, and normalized JSON (when available)
- `*.templates.json`: Learned templates with metadata

## Example

Input log:
```
Jan 1 00:00:24 10.47.0.33 dot1x-proc:1[8869]: <520002> <8869> <ERRS> Authentication timeout
```

Template learned:
```
<ts> <ip> dot1x-proc:<proc_id>[<pid>]: <id1> <id2> <level> Authentication timeout
```

Constants identified: `dot1x-proc:`, `Authentication timeout`
Variables identified: `ts`, `ip`, `proc_id`, `pid`, `id1`, `id2`, `level`

## Research Background

This implements a semantic-aware approach to log parsing:
1. LLM agents understand log semantics, not just patterns
2. Distinguishes event structure from instance data
3. Incrementally learns templates
4. Applies deterministic validation checks

## License

MIT
