-- Supabase SQL Editor: execute uma vez no projeto de destino.
-- O modelo paraphrase-multilingual-MiniLM-L12-v2 produz vetores com 384 dimensões.
-- A migration respeita o schema em que a extensão vector já está instalada.

create schema if not exists extensions;

do $$
declare
    vector_schema text;
begin
    select n.nspname
      into vector_schema
      from pg_extension e
      join pg_namespace n on n.oid = e.extnamespace
     where e.extname = 'vector';

    if vector_schema is null then
        execute 'create extension vector with schema extensions';
        vector_schema := 'extensions';
    end if;

    execute format(
        'create table if not exists public.business_knowledge_vectors (
            id text primary key,
            content text not null,
            metadata jsonb not null default ''{}''::jsonb,
            embedding %I.vector(384) not null
        )',
        vector_schema
    );

    execute format(
        'create index if not exists business_knowledge_vectors_embedding_hnsw
         on public.business_knowledge_vectors
         using hnsw (embedding %I.vector_cosine_ops)',
        vector_schema
    );
end;
$$;

create index if not exists business_knowledge_vectors_metadata_gin
    on public.business_knowledge_vectors
    using gin (metadata jsonb_path_ops);

select
    e.extname,
    e.extversion,
    n.nspname as extension_schema
from pg_extension e
join pg_namespace n on n.oid = e.extnamespace
where e.extname = 'vector';
