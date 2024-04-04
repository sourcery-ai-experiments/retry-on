# Advanced Retry Mechanism Python Package

This library offers sophisticated retry logic for both asynchronous and synchronous functions in Python, enabling developers to handle transient failures with customizable retry strategies, including controlled flow, fixed intervals, exponential backoff, and custom sequences. It integrates seamlessly with Python's asyncio and concurrent.futures for robust concurrency management and supports callback functions for additional processing upon retry.

## Features

- **Multiple Retry Patterns**: Supports controlled flow, fixed, exponential, and custom sequence retry patterns.
- **Concurrency Management**: Integrates with asyncio and ThreadPoolExecutor for managing concurrent tasks.
- **Customizable Backoff Strategies**: Configure initial delay, maximum delay, jitter, and other parameters to tailor the retry mechanism to your needs.
- **Callback Support**: Execute callback functions upon retry, allowing for logging, metrics collection, or custom recovery logic.
- **Timeouts and Limits**: Enforce function execution timeouts and set maximum retry attempts to prevent infinite loops.
- **Exception Handling**: Retry on specified exceptions, enabling targeted recovery from transient errors.

## Installation

```bash
pip install retry-on
```

## Usage

### Basic Usage

To use the retry mechanism, decorate your function with `@retry` and specify the exceptions to retry on, along with any configuration parameters.

```python
from retry_on import retry

@retry(ValueError, max_retries=3, retry_pattern='exponential')
async def example_async_function():
    # Your asynchronous operation here
    pass

@retry((IOError, OSError), retry_pattern='fixed', fixed_delay=5)
def example_sync_function():
    # Your synchronous operation here
    pass
```

### Configuring Retry Patterns

This library supports various retry patterns. Below are examples of how to configure each:

- **Controlled Flow**: Distributes retries over time, respecting rate limits.
- **Fixed**: Retries occur at fixed intervals.
- **Exponential**: Delay increases exponentially between retries.
- **Custom Sequence**: Follows a user-defined sequence of delays.

### Handling Callbacks

To perform an action upon retry, such as logging or sending alerts, define a callback function and pass it using the `on_retry_callback` parameter.

```python
def on_retry(exception):
    print(f"Retry due to {exception}")

@retry(RuntimeError, on_retry_callback=on_retry)
async def task_with_callback():
    # Task logic here
    pass
```

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
