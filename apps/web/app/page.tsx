"use client";

import {
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

type DemoUser = {
  user_id: string;
  history_count: number;
};

type Item = {
  item_id: string;
  title: string | null;
  image_url: string | null;
  main_category: string | null;
  price: string | null;
  average_rating: number | null;
  score: number;
  interaction_support: number;
  strategy: string;
};

type HistoryItem = {
  item_id: string;
  title: string | null;
  image_url: string | null;
};

type FeedResponse = {
  user_id: string;
  strategy: string;
  history_count: number;
  realtime_event_count: number;
  history_examples: HistoryItem[];
  recommendations: Item[];
};

type SearchResponse = {
  user_id: string;
  query: string;
  strategy: string;
  candidate_pool_size: number;
  query_specificity: number;
  personalization_strength: number;
  weights: {
    query: number;
    long_term: number;
    realtime: number;
    behavior: number;
  };
  realtime_event_count: number;
  results: Item[];
};

type FeedbackType =
  | "like"
  | "save"
  | "not_interested";

const API =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://localhost:8000";

export default function Home() {
  const [users, setUsers] = useState<DemoUser[]>([]);
  const [selectedUser, setSelectedUser] = useState("");
  const [feed, setFeed] =
    useState<FeedResponse | null>(null);
  const [search, setSearch] =
    useState<SearchResponse | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [pendingItem, setPendingItem] =
    useState<string | null>(null);
  const [notice, setNotice] = useState("");

  useEffect(() => {
    fetch(`${API}/api/v1/demo-users?limit=20`)
      .then((response) => response.json())
      .then((data) => {
        setUsers(data.users ?? []);

        if (data.users?.length) {
          setSelectedUser(data.users[0].user_id);
        }
      });
  }, []);

  const loadFeed = useCallback(async () => {
    if (!selectedUser) return;

    setLoading(true);

    try {
      const response = await fetch(
        `${API}/api/v1/feed/${selectedUser}?limit=24`,
        { cache: "no-store" },
      );
      const data = await response.json();

      setFeed(data);
      setSearch(null);
    } finally {
      setLoading(false);
    }
  }, [selectedUser]);

  useEffect(() => {
    loadFeed();
  }, [loadFeed]);

  const selected = useMemo(
    () =>
      users.find(
        (user) => user.user_id === selectedUser,
      ),
    [users, selectedUser],
  );

  async function runSearch(searchQuery: string) {
    if (!selectedUser || !searchQuery.trim()) {
      return;
    }

    setLoading(true);
    setNotice("");

    try {
      const params = new URLSearchParams({
        user_id: selectedUser,
        q: searchQuery.trim(),
        limit: "24",
      });

      const response = await fetch(
        `${API}/api/v1/discovery/search?${params.toString()}`,
        { cache: "no-store" },
      );

      if (!response.ok) {
        throw new Error("Search failed");
      }

      setSearch(await response.json());
    } catch {
      setNotice(
        "Search failed. Check the API terminal.",
      );
    } finally {
      setLoading(false);
    }
  }

  function submitSearch(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();
    runSearch(query);
  }

  async function reloadCurrentView() {
    if (search) {
      await runSearch(search.query);
    } else {
      await loadFeed();
    }
  }

  async function sendFeedback(
    item: Item,
    eventType: FeedbackType,
  ) {
    if (!selectedUser || pendingItem) return;

    setPendingItem(item.item_id);
    setNotice("");

    try {
      const response = await fetch(
        `${API}/api/v1/interactions`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            user_id: selectedUser,
            item_id: item.item_id,
            event_type: eventType,
          }),
        },
      );

      if (!response.ok) {
        throw new Error("Could not save feedback");
      }

      const label = {
        like: "Liked",
        save: "Saved",
        not_interested: "Hidden",
      }[eventType];

      setNotice(`${label}. Updating results…`);
      await reloadCurrentView();
    } catch {
      setNotice(
        "Could not update recommendations.",
      );
    } finally {
      setPendingItem(null);
    }
  }

  const visibleItems =
    search?.results ??
    feed?.recommendations ??
    [];

  return (
    <main>
      <header className="topbar">
        <div>
          <div className="eyebrow">
            AI-powered visual discovery
          </div>
          <h1>InspireRank</h1>
        </div>

        <div className="userPicker">
          <label htmlFor="user">
            Demo user
          </label>

          <select
            id="user"
            value={selectedUser}
            onChange={(event) => {
              setSelectedUser(
                event.target.value,
              );
              setNotice("");
              setSearch(null);
            }}
          >
            {users.map((user) => (
              <option
                key={user.user_id}
                value={user.user_id}
              >
                {user.user_id.slice(0, 12)}
                … · {user.history_count} interactions
              </option>
            ))}
          </select>
        </div>
      </header>

      <section className="hero">
        <div>
          <p className="heroLabel">
            Personalized discovery
          </p>

          <h2>
            Find what matters,
            not just what matches.
          </h2>

          <p>
            Search by meaning, retrieve relevant
            candidates, then personalize and diversify
            the final ranking.
          </p>

          <form
            className="searchBar"
            onSubmit={submitSearch}
          >
            <input
              value={query}
              onChange={(event) =>
                setQuery(event.target.value)
              }
              placeholder="Try: craft supplies"
              aria-label="Semantic search"
            />

            <button
              type="submit"
              disabled={!query.trim() || loading}
            >
              Search
            </button>
          </form>

          {search ? (
            <>
              <button
                className="backButton"
                onClick={() => {
                  setQuery("");
                  loadFeed();
                }}
              >
                ← Back to personalized feed
              </button>

              <div className="searchDiagnostics">
                <span>
                  {search.candidate_pool_size} semantic candidates
                </span>
                <span>
                  Query specificity{" "}
                  {Math.round(
                    search.query_specificity * 100,
                  )}
                  %
                </span>
                <span>
                  Personalization{" "}
                  {Math.round(
                    search.personalization_strength *
                      100,
                  )}
                  %
                </span>
              </div>
            </>
          ) : null}
        </div>

        <div className="metricGrid">
          <div className="metricCard">
            <span>History</span>
            <strong>
              {selected?.history_count ?? "—"}
            </strong>
            <small>training interactions</small>
          </div>

          <div className="metricCard">
            <span>Live signals</span>
            <strong>
              {search?.realtime_event_count ??
                feed?.realtime_event_count ??
                0}
            </strong>
            <small>recent feedback events</small>
          </div>
        </div>
      </section>

      {!search &&
      feed?.history_examples?.length ? (
        <section>
          <div className="sectionHeading">
            <div>
              <span className="eyebrow">
                Taste profile
              </span>
              <h3>Recent signals</h3>
            </div>
          </div>

          <div className="historyRow">
            {feed.history_examples.map(
              (item) => (
                <div
                  className="historyChip"
                  key={item.item_id}
                >
                  {item.image_url ? (
                    <img
                      src={item.image_url}
                      alt=""
                    />
                  ) : (
                    <div className="historyPlaceholder" />
                  )}

                  <span>
                    {item.title ??
                      "Untitled item"}
                  </span>
                </div>
              ),
            )}
          </div>
        </section>
      ) : null}

      <section>
        <div className="sectionHeading">
          <div>
            <span className="eyebrow">
              {search
                ? "Two-stage personalized search"
                : "For you"}
            </span>

            <h3>
              {search
                ? `Results for “${search.query}”`
                : "Recommended inspirations"}
            </h3>
          </div>

          <div className="feedStatus">
            {notice ? (
              <span className="notice">
                {notice}
              </span>
            ) : null}

            <span className="strategy">
              {search?.strategy ??
                feed?.strategy ??
                ""}
            </span>
          </div>
        </div>

        {loading ? (
          <div className="loading">
            {search
              ? "Retrieving and reranking candidates…"
              : "Building personalized feed…"}
          </div>
        ) : (
          <div className="masonry">
            {visibleItems.map((item) => (
              <article
                className="card"
                key={item.item_id}
              >
                <div className="imageWrap">
                  {item.image_url ? (
                    <img
                      src={item.image_url}
                      alt={item.title ?? ""}
                    />
                  ) : (
                    <div className="imagePlaceholder">
                      No image
                    </div>
                  )}

                  <span className="strategyBadge">
                    {item.strategy ===
                    "semantic+behavioral"
                      ? "Hybrid"
                      : "Semantic"}
                  </span>
                </div>

                <div className="cardBody">
                  <h4>
                    {item.title ??
                      "Untitled item"}
                  </h4>

                  <div className="meta">
                    {item.average_rating ? (
                      <span>
                        ★{" "}
                        {item.average_rating.toFixed(
                          1,
                        )}
                      </span>
                    ) : null}

                    <span>
                      {item.interaction_support} interactions
                    </span>
                  </div>

                  <div className="actions">
                    <button
                      disabled={
                        pendingItem === item.item_id
                      }
                      onClick={() =>
                        sendFeedback(item, "like")
                      }
                    >
                      ♡ Like
                    </button>

                    <button
                      disabled={
                        pendingItem === item.item_id
                      }
                      onClick={() =>
                        sendFeedback(item, "save")
                      }
                    >
                      + Save
                    </button>

                    <button
                      className="hideAction"
                      disabled={
                        pendingItem === item.item_id
                      }
                      onClick={() =>
                        sendFeedback(
                          item,
                          "not_interested",
                        )
                      }
                    >
                      × Hide
                    </button>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
