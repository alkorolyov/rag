-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for embeddings
CREATE SCHEMA IF NOT EXISTS embeddings;

-- Grant permissions
GRANT ALL ON SCHEMA embeddings TO CURRENT_USER;

-- Verify extension installation
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';