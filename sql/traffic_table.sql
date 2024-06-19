SELECT "DateTime", "Junction", "Vehicles", "ID" 
FROM public.traffic

SELECT * FROM traffic LIMIT 10

INSERT INTO traffic ("DateTime", "Junction", "Vehicles", "ID")
VALUES 
('2017-07-05 19:00:00', 1, 60, 20170705119),
('2017-07-05 20:00:00', 1, 50, 20170705120),
('2017-07-06 19:00:00', 1, 40, 20170706119),
('2017-07-06 20:00:00', 1, 65, 20170706120),
('2017-07-07 19:00:00', 1, 55, 20170707119),
('2017-07-07 20:00:00', 1, 45, 20170707120),
('2017-07-08 19:00:00', 1, 75, 20170708119),
('2017-07-08 20:00:00', 1, 65, 20170708120),
('2017-07-09 19:00:00', 1, 55, 20170709119),
('2017-07-09 20:00:00', 1, 60, 20170709120),
('2017-07-10 17:00:00', 1, 100, 20170710117),
('2017-07-10 18:00:00', 1, 130, 20170710118),
('2017-07-11 17:00:00', 1, 50, 20170711117),
('2017-07-11 18:00:00', 1, 70, 20170711118),
('2017-07-12 17:00:00', 1, 90, 20170712117),
('2017-07-12 18:00:00', 1, 80, 20170712118),
('2017-07-13 17:00:00', 1, 120, 20170713117),
('2017-07-13 18:00:00', 1, 150, 20170713118)

SELECT * FROM traffic WHERE "ID" = 20170705119 
OR "ID" = 20170705120

DELETE FROM traffic WHERE "ID" = 20170705119 
OR "ID" = 20170705120

DELETE FROM traffic