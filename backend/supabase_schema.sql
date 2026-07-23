-- -------------------------------------------------------------
-- PrepAI — Supabase Table & RLS Setup Script
-- Run this script in the Supabase SQL Editor (Database -> SQL Editor)
-- -------------------------------------------------------------

-- 1. Create the USERS table (Stores user name & email immediately on start)
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Create the SESSIONS table (Stores full mock test results)
CREATE TABLE IF NOT EXISTS public.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    user_name TEXT,
    topics JSONB NOT NULL,
    difficulty TEXT NOT NULL,
    total_questions INT NOT NULL DEFAULT 0,
    correct INT NOT NULL DEFAULT 0,
    partial INT NOT NULL DEFAULT 0,
    wrong INT NOT NULL DEFAULT 0,
    score_pct INT NOT NULL DEFAULT 0,
    duration_seconds INT,
    results JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users (email);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON public.sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON public.sessions (created_at DESC);

-- 4. Enable Row Level Security (RLS)
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;

-- 5. Create RLS policies to allow inserts/reads/updates using both Anon and Service Role keys
DROP POLICY IF EXISTS "Allow public access to users" ON public.users;
CREATE POLICY "Allow public access to users" 
ON public.users 
FOR ALL 
USING (true) 
WITH CHECK (true);

DROP POLICY IF EXISTS "Allow public insert to sessions" ON public.sessions;
CREATE POLICY "Allow public insert to sessions" 
ON public.sessions 
FOR INSERT 
WITH CHECK (true);

DROP POLICY IF EXISTS "Allow public select from sessions" ON public.sessions;
CREATE POLICY "Allow public select from sessions" 
ON public.sessions 
FOR SELECT 
USING (true);
