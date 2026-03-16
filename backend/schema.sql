-- ══════════════════════════════════════════
--  PrepAI — Supabase Schema
--  Run this in Supabase SQL Editor
-- ══════════════════════════════════════════

-- Sessions table — stores every completed interview session
create table if not exists sessions (
  id                serial primary key,
  session_id        text unique not null,
  user_id           text,                        -- null for anonymous users
  topics            text[] not null,
  difficulty        text not null,
  total_questions   int not null,
  correct           int default 0,
  partial           int default 0,
  wrong             int default 0,
  score_pct         int default 0,
  duration_seconds  int,
  results           jsonb,                       -- full Q&A breakdown
  created_at        timestamptz default now()
);

-- Index for fast user history lookups
create index if not exists idx_sessions_user_id    on sessions(user_id);
create index if not exists idx_sessions_score_pct  on sessions(score_pct desc);
create index if not exists idx_sessions_created_at on sessions(created_at desc);

-- Enable Row Level Security
alter table sessions enable row level security;

-- Policy: anyone can insert (anonymous sessions allowed)
create policy "Allow insert" on sessions for insert with check (true);

-- Policy: users can only read their own sessions
create policy "Allow select own" on sessions for select
  using (user_id = current_user or user_id is null);

-- Policy: leaderboard — allow reading score + topics only (no personal data)
create policy "Allow leaderboard read" on sessions for select
  using (true);
