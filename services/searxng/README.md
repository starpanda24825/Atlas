# SearXNG tuning for Atlas

Atlas reaches its SearXNG instance through
[`core/config.py`](../../core/config.py) → `SEARXNG_URL`
(default `http://localhost:8888`, the port
[`~/searxng/docker-compose.yml`](https://docs.searxng.org/) maps to 8080).

`settings.yml` here is the **canonical** copy of the instance config.
`~/searxng/searxng/settings.yml` is the **live** file the container bind-mounts
at `/etc/searxng` — the two must be kept in sync by re-running the apply step.

## Apply the settings and restart

Run these from the Atlas repo root. The first command reads the existing
`secret_key` out of the live file so it is never committed to git.

```bash
SECRET=$(docker compose -f ~/searxng/docker-compose.yml exec -T searxng \
  sed -n 's/^  secret_key: "\(.*\)"/\1/p' /etc/searxng/settings.yml) && \
sed "s|__SECRET_KEY__|$SECRET|" services/searxng/settings.yml | \
docker compose -f ~/searxng/docker-compose.yml exec -T searxng \
  sh -c 'cat > /etc/searxng/settings.yml' && \
docker compose -f ~/searxng/docker-compose.yml restart
```

`cat >` truncates the existing file, so its ownership (`searxng:searxng`) is
preserved and the container keeps read access. The container's entrypoint also
re-`chown`s the config directory on every start.

## Verify

```bash
# effective values, read from inside the container
docker compose -f ~/searxng/docker-compose.yml exec -T searxng \
  sh -c 'cd /usr/local/searxng && ./.venv/bin/python -c "
from searx import get_setting
print(get_setting(\"outgoing.request_timeout\"))
print(get_setting(\"search.suspended_times\"))"'

# is wikipedia in the results list now?
python3 -c "
import json, urllib.request
d = json.loads(urllib.request.urlopen(
    'http://localhost:8888/search?q=nvidia+earnings&format=json').read())
from collections import Counter
print(Counter(r['engine'] for r in d['results']))"
```

## Notes on the engine list

`google`, `bing` and `yahoo` ship as `disabled: true` in this SearXNG build, so
enabling them is a real change — it affects searches that do **not** pass an
explicit `engines=` list, including the web UI at <http://localhost:8888>.
Atlas passes an explicit list (see `SEARXNG_ENGINES` in `core/config.py`), which
SearXNG honours even for disabled engines.

`wikipedia` previously returned zero results for every query because its
`display_type` defaulted to `["infobox"]`: hits went to the JSON `infoboxes`
array instead of `results`. Adding `"list"` fixes it.

There is **no** `reddit` engine in SearXNG. Use `site:reddit.com` in a query
against google/bing instead. There is likewise no Yahoo Finance engine; Atlas
gets live quotes from `yfinance` in `search/quick_search.py`.
