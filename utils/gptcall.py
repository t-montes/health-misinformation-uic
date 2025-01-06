from openai import OpenAI as OpenAiClient
from openai._exceptions import ContentFilterFinishReasonError, RateLimitError, AuthenticationError
import time

def request(prompt, model='gpt-4o-mini', api_key=None, system_message=None, prev_msgs=[], max_retries=3, retry_delay=60, try_count=0, max_tokens=4096, temperature=0, seed=None, track_usage=False):
    gpt_client = OpenAiClient(api_key=api_key) # OPENAI_API_KEY
    msgs = prev_msgs.copy()
    if system_message:
        msgs.append({"role": "system", "content": system_message})
    msgs.append({"role": "user", "content": prompt})
    try:
        response = gpt_client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed, # seed for deterministic completions, if set
            messages=msgs
        )
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "content_filter":
            raise ContentFilterFinishReasonError(response.choices[0].content_filter_results)
        rsp = response.choices[0].message.content
        if track_usage:
            return rsp, { "input": response.usage.prompt_tokens, "output": response.usage.completion_tokens }
        return rsp
    except Exception as e:
        if "Max retries exceeded with url" in f"{e}":
            if try_count >= max_retries:
                raise RateLimitError(f"Max retries exceeded with url: {e}")
            print(f"RETRYING ({try_count})...")
            time.sleep(retry_delay)
            return request(prompt, model, system_message, prev_msgs, max_retries, retry_delay, try_count+1, max_tokens, temperature, seed, track_usage)
        raise e
    except AuthenticationError as e:
        raise AuthenticationError(f"API key not valid or not provided")
