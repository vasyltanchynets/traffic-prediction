--create elt user
CREATE USER elt WITH PASSWORD 'demopass';
--grant connect
GRANT CONNECT ON DATABASE "traffic_prediction" TO elt;
--grant table permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO elt;
--permission for schema public
GRANT ALL ON SCHEMA public TO elt;
