CREATE TABLE IF NOT EXISTS public.stock
(
    id integer NOT NULL DEFAULT nextval('portfolio_id_seq'::regclass),
    ticker text COLLATE pg_catalog."default" NOT NULL,
    name text COLLATE pg_catalog."default" NOT NULL,
    price numeric,
    number_stocks numeric NOT NULL,
    CONSTRAINT portfolio_pkey PRIMARY KEY (id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.stock
    OWNER to moutz;