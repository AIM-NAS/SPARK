import http.client
import json
from urllib.parse import urlparse
print(">>> USING PATCHED api_general.py")

class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def _parse_endpoint(self):
        """
        Support both:
        1) full base url, e.g. https://dashscope.aliyuncs.com/compatible-mode/v1
        2) host only, e.g. api.deepseek.com
        """
        endpoint = self.api_endpoint.strip()

        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            parsed = urlparse(endpoint)
            host = parsed.netloc
            base_path = parsed.path.rstrip("/")
            if not base_path:
                base_path = ""
            return host, base_path

        # host-only mode
        return endpoint, ""

    def get_response(self, prompt_content):
        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,
                "messages": [
                    {"role": "user", "content": prompt_content}
                ],
            }
        )

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": "1",
        }

        response = None
        n_trial = 1

        host, base_path = self._parse_endpoint()

        # If endpoint is a full OpenAI-compatible base_url like
        # https://dashscope.aliyuncs.com/compatible-mode/v1
        # then request path should be:
        # /compatible-mode/v1/chat/completions
        #
        # If endpoint is host only like api.deepseek.com
        # then request path remains:
        # /v1/chat/completions
        if base_path:
            request_path = f"{base_path}/chat/completions"
        else:
            request_path = "/v1/chat/completions"

        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                if self.debug_mode:
                    print(f"[LLM] host={host}")
                    print(f"[LLM] path={request_path}")
                    print(f"[LLM] model={self.model_LLM}")

                conn = http.client.HTTPSConnection(host, timeout=120)
                conn.request("POST", request_path, payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()

                if self.debug_mode:
                    print(f"[LLM] status={res.status}")
                    print(f"[LLM] raw={data[:500]}")

                json_data = json.loads(data)

                # Helpful error visibility
                if "choices" not in json_data:
                    if self.debug_mode:
                        print("[LLM] response json:", json_data)
                    return None

                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in API. Restarting the process... {e}")
                continue

        return response