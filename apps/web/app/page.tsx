"use client";

import { useEffect, useMemo, useState } from "react";

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
  history_examples: HistoryItem[];
  recommendations: Item[];
};

const API =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function Home() {
  const [users, setUsers] = useState<DemoUser[]>([]);
  const [selectedUser, setSelectedUser] = useState("");
  const [feed, setFeed] = useState<FeedResponse | null>(null);
  const [loading, setLoading] = useState(false);

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

  useEffect(() => {
    if (!selectedUser) return;

    setLoading(true);

    fetch(`${API}/api/v1/feed/${selectedUser}?limit=24`)
      .then((response) => response.json())
      .then((data) => setFeed(data))
      .finally(() => setLoading(false));
  }, [selectedUser]);

  const selected = useMemo(
    () => users.find((user) => user.user_id === selectedUser),
    [users, selectedUser],
  );

  return (
    <main>
      <header className="topbar">
        <div>
          <div className="eyebrow">AI-powered visual discovery</div>
          <h1>InspireRank</h1>
        </div>

        <div className="userPicker">
          <label htmlFor="user">Demo user</label>
          <select
            id="user"
            value={selectedUser}
            onChange={(event) => setSelectedUser(event.target.value)}
          >
            {users.map((user) => (
              <option key={user.user_id} value={user.user_id}>
                {user.user_id.slice(0, 12)}… · {user.history_count} interactions
              </option>
            ))}
          </select>
        </div>
      </header>

      <section className="hero">
        <div>
          <p className="heroLabel">Personalized for this user</p>
          <h2>Discover what they may want next.</h2>
          <p>
            Hybrid retrieval combines zero-shot semantic understanding with
            behavioral signals for well-supported items.
          </p>
        </div>

        <div className="metricCard">
          <span>History</span>
          <strong>{selected?.history_count ?? "—"}</strong>
          <small>training interactions</small>
        </div>
      </section>

      {feed?.history_examples?.length ? (
        <section className="historySection">
          <div className="sectionHeading">
            <div>
              <span className="eyebrow">Taste profile</span>
              <h3>Recent signals</h3>
            </div>
          </div>

          <div className="historyRow">
            {feed.history_examples.map((item) => (
              <div className="historyChip" key={item.item_id}>
                {item.image_url ? (
                  <img src={item.image_url} alt="" />
                ) : (
                  <div className="historyPlaceholder" />
                )}
                <span>{item.title ?? "Untitled item"}</span>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <section>
        <div className="sectionHeading">
          <div>
            <span className="eyebrow">For you</span>
            <h3>Recommended inspirations</h3>
          </div>
          {feed ? (
            <span className="strategy">{feed.strategy}</span>
          ) : null}
        </div>

        {loading ? (
          <div className="loading">Building personalized feed…</div>
        ) : (
          <div className="masonry">
            {feed?.recommendations?.map((item) => (
              <article className="card" key={item.item_id}>
                <div className="imageWrap">
                  {item.image_url ? (
                    <img src={item.image_url} alt={item.title ?? ""} />
                  ) : (
                    <div className="imagePlaceholder">No image</div>
                  )}

                  <span className="strategyBadge">
                    {item.strategy === "semantic+behavioral"
                      ? "Hybrid"
                      : "Semantic"}
                  </span>
                </div>

                <div className="cardBody">
                  <h4>{item.title ?? "Untitled item"}</h4>

                  <div className="meta">
                    {item.average_rating ? (
                      <span>★ {item.average_rating.toFixed(1)}</span>
                    ) : null}
                    <span>{item.interaction_support} interactions</span>
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
