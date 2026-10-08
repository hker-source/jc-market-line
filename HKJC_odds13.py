import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from playwright.sync_api import sync_playwright


class HKJCGraphQLClient:
    """HKJC GraphQL API client using Playwright to bypass whitelist verification.

    HKJC's API rejects direct HTTP requests (WHITELIST_ERROR).  By loading the
    site in a headless Chromium browser, the website's own JavaScript makes the
    GraphQL calls with proper dynamic tokens/cookies.  We intercept those
    responses to obtain the data.
    """

    BASE_URL = "https://info.cld.hkjc.com/graphql/base/"
    SITE_URL = "https://bet.hkjc.com/"

    # odds_type -> the HKJC sub-page whose own JS auto-fires that odds_type's
    # matchList query. Mirrors enum_code_table.csv's page_url column.
    # Visiting the right page first gets the correct Referer + cookies/tokens
    # for that odds_type, which our own page.evaluate() fetch cannot spoof
    # (Referer is a forbidden header for script-set fetch()).
    ODDS_TYPE_PAGES = {
        "HAD": SITE_URL, "SGA": SITE_URL,
        "NGS": SITE_URL + "en/football/had", "ENT": SITE_URL + "en/football/had",
        "NTS": SITE_URL + "en/football/had", "EHA": SITE_URL + "en/football/had",
        "CHH": SITE_URL + "en/football/teamchl", "CEH": SITE_URL + "en/football/teamchl",
        "CHA": SITE_URL + "en/football/teamchl", "CEA": SITE_URL + "en/football/teamchl",
        "ECH": SITE_URL + "en/football/chl", "CHL": SITE_URL + "en/football/chl",
        "HIL": SITE_URL + "en/football/hil", "EHL": SITE_URL + "en/football/hil",
        "ECS": SITE_URL + "en/football/crs", "CRS": SITE_URL + "en/football/crs",
        "FTS": SITE_URL + "en/football/fts",
        "HLA": SITE_URL + "en/football/teamhil", "ELH": SITE_URL + "en/football/teamhil",
        "HLH": SITE_URL + "en/football/teamhil", "ELA": SITE_URL + "en/football/teamhil",
        "EDC": SITE_URL + "en/football/hdc", "HDC": SITE_URL + "en/football/hdc",
        "CHD": SITE_URL + "en/football/chd", "ECD": SITE_URL + "en/football/chd",
    }

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._graphql_responses = []
        self._graphql_requests = []

    def __enter__(self):
        self._graphql_responses.clear()
        self._graphql_requests.clear()
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-web-security", "--disable-features=IsolateOrigins,site-per-process"],
        )
        self._context = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            )
        )
        self._page = self._context.new_page()

        def _capture_response(response):
            if "graphql" in response.url:
                self._graphql_responses.append(response)

        def _capture_request(request):
            if "graphql" in request.url:
                self._graphql_requests.append(request)

        self._page.on("response", _capture_response)
        self._page.on("request", _capture_request)

        self._page.goto(self.SITE_URL, wait_until="networkidle")
        self._page.wait_for_timeout(8000)
        self._current_page_url = self.SITE_URL  # tracked across polls, not reset per call
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def clear_buffer(self):
        self._graphql_responses.clear()
        self._graphql_requests.clear()

    def save_raw(self, data: Dict[str, Any], odds_type: str, label: str = "") -> Path:
        raw_dir = Path(__file__).parent / "raw" / odds_type
        raw_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        name_part = label.replace(" ", "_") if label else "data"
        path = raw_dir / f"{name_part}_{ts}.json"
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Raw saved: {path}")
        return path

    def _graph_query(self, query: str, operation_name: str,
                     variables: Dict[str, Any],
                     target_odds_types: Optional[List[str]] = None,
                     retries: int = 1) -> Dict[str, Any]:
        for attempt in range(retries + 1):
            self._page.wait_for_timeout(5000)

            if target_odds_types:
                cached_seen_types = set()
                for resp in self._graphql_responses:
                    try:
                        data = resp.json()
                        matches = data.get("data", {}).get("matches")
                        if matches is None:
                            continue
                        for m in matches:
                            for pool in (m.get("foPools") or []):
                                cached_seen_types.add(pool.get("oddsType"))
                                if pool.get("oddsType") in target_odds_types:
                                    print(f"Found {target_odds_types} in cached response")
                                    return data
                    except Exception:
                        pass
                print(f"No {target_odds_types} in cache (cache had oddsTypes = {cached_seen_types or '(none)'}), "
                      f"trying active fetch...")
            else:
                for resp in self._graphql_responses:
                    try:
                        data = resp.json()
                        if data.get("data", {}).get("matches") is not None:
                            print("Found matches data in cached response")
                            return data
                    except Exception:
                        pass

            if self._graphql_requests:
                req = None
                for r in reversed(self._graphql_requests):
                    try:
                        body = r.post_data
                        if body and '"matchList"' in body:
                            req = r
                            break
                    except Exception:
                        pass
                if req is None:
                    req = self._graphql_requests[-1]
                # Different odds_types can route through different backend
                # hosts (e.g. info.cld.hkjc.com for HAD/SGA vs
                # consvc.hkjc.com/JCBW/api/graph for HDC/EDC/CHD/ECD) — use
                # the captured request's own URL rather than a hardcoded
                # BASE_URL so we always POST to the right endpoint.
                target_url = req.url
                # Exclude hop-by-hop/forbidden headers plus browser-internal
                # ones (priority, sec-fetch-*, sec-ch-*, HTTP/2 pseudo-headers)
                # that aren't safe to replay from script and can trip CORS
                # preflight (observed: "priority" rejected by
                # Access-Control-Allow-Headers on consvc.hkjc.com).
                _skip_prefixes = ("sec-", ":")
                _skip_exact = {"content-length", "content-encoding", "host", "priority",
                                "origin", "referer", "cookie"}
                extra_headers = {
                    k: v for k, v in req.headers.items()
                    if k.lower() not in _skip_exact and not k.lower().startswith(_skip_prefixes)
                }
                payload = json.dumps({
                    "operationName": operation_name,
                    "query": query,
                    "variables": variables,
                })
                result = self._page.evaluate(
                    """([p, url, hdrs]) => {
    const headers = { ...hdrs };
    return fetch(url, {
        method: "POST",
        headers: headers,
        body: p,
        credentials: "include",
    }).then(r => r.text()).then(t => {
        try { return JSON.parse(t); }
        catch(e) { return { __raw: t }; }
    });
}""",
                    (payload, target_url, extra_headers),
                )

                if isinstance(result, dict) and "__raw" not in result:
                    if result.get("errors"):
                        # GraphQL returns HTTP 200 even for validation errors
                        # (bad enum value, wrong field for this endpoint's
                        # schema, etc.) -- we were silently treating that the
                        # same as "no data", which hid the real reason.
                        print(f"Active fetch GraphQL errors for {target_odds_types}: {result['errors']}")
                    if target_odds_types:
                        matches = result.get("data", {}).get("matches") or []
                        seen_types = set()
                        for m in matches:
                            for pool in (m.get("foPools") or []):
                                seen_types.add(pool.get("oddsType"))
                                if pool.get("oddsType") in target_odds_types:
                                    print(f"Found {target_odds_types} in active fetch")
                                    return result
                        print(f"Active fetch ({target_url}): got {len(matches)} matches, "
                              f"oddsTypes present = {seen_types or '(none)'}, "
                              f"wanted {target_odds_types} -- not present, likely no live market right now")
                    elif result.get("data", {}).get("matches") is not None:
                        return result

                if isinstance(result, dict) and "__raw" in result:
                    raw = result["__raw"]
                    print(f"Fetch returned non-JSON ({len(raw)} chars), first 200: {raw[:200]}")

            if attempt < retries:
                print(f"Attempt {attempt + 1} failed for {target_odds_types}, retrying in 3s...")
                self._page.wait_for_timeout(3000)

        print(f"No data found for {target_odds_types} after {retries + 1} attempt(s)")
        return {}

    def send_basic_match_list_request(self) -> Dict[str, Any]:
        query = """query matchList($startIndex: Int, $endIndex: Int, $startDate: String, $endDate: String, $matchIds: [String], $tournIds: [String], $fbOddsTypes: [FBOddsType]!, $fbOddsTypesM: [FBOddsType]!, $inplayOnly: Boolean, $featuredMatchesOnly: Boolean, $frontEndIds: [String], $earlySettlementOnly: Boolean, $showAllMatch: Boolean) {
  matches(startIndex: $startIndex, endIndex: $endIndex, startDate: $startDate, endDate: $endDate, matchIds: $matchIds, tournIds: $tournIds, fbOddsTypes: $fbOddsTypesM, inplayOnly: $inplayOnly, featuredMatchesOnly: $featuredMatchesOnly, frontEndIds: $frontEndIds, earlySettlementOnly: $earlySettlementOnly, showAllMatch: $showAllMatch) {
    id
    frontEndId
    matchDate
    kickOffTime
    status
    homeTeam { id name_en name_ch }
    awayTeam { id name_en name_ch }
    tournament { id frontEndId name_ch name_en }
    venue { id name_ch name_en }
    runningResult { homeScore awayScore homeCorner awayCorner }
  }
}"""

        variables = {
            "fbOddsTypes": [],
            "fbOddsTypesM": [],
            "inplayOnly": False,
            "featuredMatchesOnly": False,
            "startDate": None,
            "endDate": None,
            "tournIds": None,
            "matchIds": None,
            "startIndex": 1,
            "endIndex": 60,
            "frontEndIds": None,
            "earlySettlementOnly": False,
            "showAllMatch": False,
        }

        return self._graph_query(query, "matchList", variables)

    _DETAILED_QUERY = """query matchList($startIndex: Int, $endIndex: Int, $startDate: String, $endDate: String, $matchIds: [String], $tournIds: [String], $fbOddsTypes: [FBOddsType]!, $fbOddsTypesM: [FBOddsType]!, $inplayOnly: Boolean, $featuredMatchesOnly: Boolean, $frontEndIds: [String], $earlySettlementOnly: Boolean, $showAllMatch: Boolean) {
  matches(startIndex: $startIndex, endIndex: $endIndex, startDate: $startDate, endDate: $endDate, matchIds: $matchIds, tournIds: $tournIds, fbOddsTypes: $fbOddsTypesM, inplayOnly: $inplayOnly, featuredMatchesOnly: $featuredMatchesOnly, frontEndIds: $frontEndIds, earlySettlementOnly: $earlySettlementOnly, showAllMatch: $showAllMatch) {
    id
    frontEndId
    matchDate
    kickOffTime
    status
    homeTeam { id name_en name_ch }
    awayTeam { id name_en name_ch }
    tournament { id frontEndId name_ch name_en }
    venue { id name_ch name_en }
    runningResult { homeScore awayScore homeCorner awayCorner }
    foPools(fbOddsTypes: $fbOddsTypes) {
      id status oddsType instNo inplay name_ch name_en updateAt
      lines {
        lineId status condition main
        combinations {
          combId str status currentOdds
          selections { selId str name_ch name_en }
        }
      }
    }
  }
}"""

    @staticmethod
    def _chunk_ranges(start_index: int, end_index: int, chunk_size: int = 60):
        chunks = []
        s = start_index
        while s <= end_index:
            e = min(s + chunk_size - 1, end_index)
            chunks.append((s, e))
            s = e + 1
        return chunks

    def _single_detailed_request(
        self,
        odds_types: List[str],
        start_index: int,
        end_index: int,
        inplay_only: bool,
        featured_only: bool,
        start_date: Optional[str],
        end_date: Optional[str],
        result_only: bool = False,
        show_all_match: bool = False,
    ) -> Dict[str, Any]:
        variables = {
            "fbOddsTypes": odds_types,
            "fbOddsTypesM": odds_types,
            "inplayOnly": inplay_only,
            "featuredMatchesOnly": featured_only,
            "startDate": start_date,
            "endDate": end_date,
            "tournIds": None,
            "matchIds": None,
            "startIndex": start_index,
            "endIndex": end_index,
            "frontEndIds": None,
            "earlySettlementOnly": False,
            "showAllMatch": show_all_match,
        }
        query = self._DETAILED_QUERY
        if result_only:
            # resultOnly:true returns only SETTLED pools; each combination then
            # carries status WIN/LOSE and winOrd, which is the settlement feed.
            # Applied ONLY on request so live pre-match polling is unchanged.
            query = query.replace(
                "foPools(fbOddsTypes: $fbOddsTypes)",
                "foPools(fbOddsTypes: $fbOddsTypes, resultOnly: true)",
                1,
            )
        return self._graph_query(query, "matchList", variables, target_odds_types=odds_types)

    def send_detailed_match_list_request(
        self,
        odds_types: Optional[List[str]] = None,
        start_index: int = 1,
        end_index: int = 60,
        inplay_only: bool = False,
        featured_only: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        result_only: bool = False,
        show_all_match: bool = False,
    ) -> Dict[str, Any]:
        if odds_types is None:
            odds_types = ["HAD"]

        chunks = self._chunk_ranges(start_index, end_index, chunk_size=60)
        if len(chunks) == 1:
            return self._single_detailed_request(
                odds_types, start_index, end_index, inplay_only, featured_only,
                start_date, end_date, result_only=result_only, show_all_match=show_all_match,
            )

        print(f"Range {start_index}-{end_index} exceeds 60; auto-chunking into {len(chunks)} requests")
        merged_matches: List[Dict[str, Any]] = []
        seen_ids = set()
        for i, (s, e) in enumerate(chunks):
            print(f"  chunk {i + 1}/{len(chunks)}: {s}-{e}")
            chunk_data = self._single_detailed_request(
                odds_types, s, e, inplay_only, featured_only, start_date, end_date,
                result_only=result_only, show_all_match=show_all_match,
            )
            chunk_matches = (chunk_data.get("data") or {}).get("matches") or []
            for m in chunk_matches:
                mid = m.get("id")
                if mid is not None and mid in seen_ids:
                    continue
                if mid is not None:
                    seen_ids.add(mid)
                merged_matches.append(m)
            if i < len(chunks) - 1:
                self._page.wait_for_timeout(1500)

        return {"data": {"matches": merged_matches}}

    def fetch_multiple_odds_types(
        self,
        odds_types_list: List[str],
        start_index: int = 1,
        end_index: int = 60,
        inplay_only: bool = False,
        featured_only: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_raw: bool = False,
        navigate_first: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        # Each call = one poll. Clear the response/request cache first -- this
        # buffer otherwise only gets cleared once, at __enter__, so a
        # long-running poller (same client kept alive across many polls) would
        # keep matching the OLDEST cached response for each odds_type forever
        # (cache-scan does for resp in self._graphql_responses: ... return on
        # first match -- oldest-first, since append() adds to the end). That
        # silently re-ingests poll #1's data every subsequent poll. Clearing
        # here forces every poll to see only its own freshly captured traffic.
        self.clear_buffer()
        results = {}
        last_url = getattr(self, "_current_page_url", self.SITE_URL)
        for odds_type in odds_types_list:
            target_url = self.ODDS_TYPE_PAGES.get(odds_type.upper())
            if navigate_first and target_url and target_url != last_url:
                print(f"Navigating to {target_url} (for {odds_type})")
                self._page.goto(target_url, wait_until="networkidle")
                self._page.wait_for_timeout(6000)
                last_url = target_url
                self._current_page_url = target_url
            elif navigate_first and not target_url:
                print(f"No page mapping for {odds_type} — staying on {last_url}, relying on active fetch")

            print(f"\n--- Fetching {odds_type} ---")
            data = self.send_detailed_match_list_request(
                odds_types=[odds_type],
                start_index=start_index,
                end_index=end_index,
                inplay_only=inplay_only,
                featured_only=featured_only,
                start_date=start_date,
                end_date=end_date,
            )
            if data:
                results[odds_type] = data
                if save_raw:
                    self.save_raw(data, odds_type, f"{start_index}-{end_index}")
            if odds_type != odds_types_list[-1]:
                time.sleep(2)
        return results

    def fetch_settlement(
        self,
        odds_types: List[str],
        start_index: int = 1,
        end_index: int = 60,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_raw: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch SETTLED (ended) matches with foPools(resultOnly:true).

        Mirrors fetch_multiple_odds_types but requests settlement: every
        combination returns status WIN/LOSE + winOrd, and lines carry
        lineId/condition (unlike the standalone matchResultDetails payload), so
        the result joins straight onto odds_snapshot/opening_odds on
        (match_id, odds_type, pool_id, line_id, comb_id).
        """
        self.clear_buffer()
        results: Dict[str, Dict[str, Any]] = {}
        last_url = getattr(self, "_current_page_url", self.SITE_URL)
        for odds_type in odds_types:
            target_url = self.ODDS_TYPE_PAGES.get(odds_type.upper())
            if target_url and target_url != last_url:
                print(f"Navigating to {target_url} (for {odds_type} settlement)")
                self._page.goto(target_url, wait_until="networkidle")
                self._page.wait_for_timeout(6000)
                last_url = target_url
                self._current_page_url = target_url
            print(f"\n--- Fetching {odds_type} settlement ---")
            data = self.send_detailed_match_list_request(
                odds_types=[odds_type],
                start_index=start_index,
                end_index=end_index,
                start_date=start_date,
                end_date=end_date,
                result_only=True,
                show_all_match=True,
            )
            if data:
                results[odds_type] = data
                if save_raw:
                    self.save_raw(data, odds_type, f"settled_{start_index}-{end_index}")
            if odds_type != odds_types[-1]:
                time.sleep(2)
        return results

    def parse_had_odds(self, api_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        data_content = api_data.get("data") or {}
        matches = data_content.get("matches") or []
        parsed_results = []

        for match in matches:
            info = {
                "match_id": match.get("frontEndId"),
                "league": match.get("tournament", {}).get("name_ch"),
                "home_team": match.get("homeTeam", {}).get("name_ch"),
                "away_team": match.get("awayTeam", {}).get("name_ch"),
                "odds": {}
            }
            for pool in (match.get("foPools") or []):
                if pool.get("oddsType") == "HAD":
                    for line in (pool.get("lines") or []):
                        for comb in (line.get("combinations") or []):
                            info["odds"][comb.get("str")] = comb.get("currentOdds")
            if info["odds"]:
                parsed_results.append(info)

        return parsed_results

    def matches_to_dataframe(self, api_data: Dict[str, Any]) -> pd.DataFrame:
        data_content = api_data.get("data") or {}
        matches = data_content.get("matches") or []
        rows = []

        for match in matches:
            running_result = match.get("runningResult") or {}
            fo_pools = match.get("foPools") or []
            venue = match.get("venue") or {}
            home_team = match.get("homeTeam") or {}
            away_team = match.get("awayTeam") or {}
            tournament = match.get("tournament") or {}

            has_normal = any(not pool.get("inplay", False) for pool in fo_pools)
            has_live = any(pool.get("inplay", False) for pool in fo_pools)

            rows.append({
                "比賽ID": match.get("id"),
                "前端ID": match.get("frontEndId"),
                "主隊": home_team.get("name_ch") or "",
                "客隊": away_team.get("name_ch") or "",
                "聯賽": tournament.get("name_ch") or "",
                "比賽日期": match.get("matchDate"),
                "開球時間": match.get("kickOffTime"),
                "狀態": match.get("status"),
                "場地": venue.get("name_ch") or "",
                "主隊比分": running_result.get("homeScore") or 0,
                "客隊比分": running_result.get("awayScore") or 0,
                "主隊角球": running_result.get("homeCorner") or 0,
                "客隊角球": running_result.get("awayCorner") or 0,
                "正常投注池": has_normal,
                "走地投注池": has_live,
                "賠率池數量": len(fo_pools),
                "更新時間": match.get("updateAt"),
            })

        return pd.DataFrame(rows)

    def odds_to_dataframe(self, api_data: Dict[str, Any]) -> pd.DataFrame:
        data_content = api_data.get("data") or {}
        matches = data_content.get("matches") or []
        rows = []

        for match in matches:
            home = (match.get("homeTeam") or {}).get("name_ch") or ""
            away = (match.get("awayTeam") or {}).get("name_ch") or ""
            match_label = f"{home} VS {away}"

            for pool in (match.get("foPools") or []):
                pool_name = pool.get("name_ch") or ""
                odds_type = pool.get("oddsType") or ""
                is_live = pool.get("inplay", False)
                pool_status = pool.get("status", "")
                pool_update_at = pool.get("updateAt")

                for line in (pool.get("lines") or []):
                    line_condition = line.get("condition") or "N/A"

                    for comb in (line.get("combinations") or []):
                        selections = comb.get("selections") or []
                        if selections:
                            option_names = " / ".join(
                                sel.get("name_ch") or sel.get("str", "") for sel in selections
                            )
                        else:
                            option_names = comb.get("str") or ""

                        rows.append({
                            "比賽": match_label,
                            "投注類型": pool_name,
                            "賠率類型": odds_type,
                            "是否走地": is_live,
                            "線路條件": line_condition,
                            "組合字串": comb.get("str") or "",
                            "選項名稱": option_names,
                            "當前賠率": comb.get("currentOdds"),
                            "組合狀態": comb.get("status") or "",
                            "提早結算": comb.get("offerEarlySettlement"),
                            "投注池狀態": pool_status or "",
                            "更新時間": pool_update_at,
                        })

        return pd.DataFrame(rows)

    def display_matches_table(self, df: pd.DataFrame) -> None:
        print("\n🏆 HKJC 比賽列表")
        print(df.to_string(index=False))

    def display_odds_table(self, df: pd.DataFrame) -> None:
        print("\n💰 HKJC 賠率信息")
        print(df.to_string(index=False))


if __name__ == "__main__":
    with HKJCGraphQLClient(headless=True) as client:
        matches_data = client.send_basic_match_list_request()
        matches_df = client.matches_to_dataframe(matches_data)
        client.display_matches_table(matches_df)

        # HAD/SGA already auto-load on the homepage; navigate_first no-ops for them
        # since ODDS_TYPE_PAGES points both at SITE_URL (same as __enter__'s load).
        results = client.fetch_multiple_odds_types(
            odds_types_list=["HAD", "SGA", "HDC", "EDC", "CHD", "ECD"],
            start_index=1,
            end_index=40,
            save_raw=True,
        )
        for odds_type, data in results.items():
            print(f"\n=== {odds_type} odds ({len(data.get('data', {}).get('matches', []))} matches) ===")
            odds_df = client.odds_to_dataframe(data)
            client.display_odds_table(odds_df)
