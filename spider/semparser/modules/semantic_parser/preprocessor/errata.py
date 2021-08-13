# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

errata = {
    'SELECT DISTINCT T1.first_name ,  T1.last_name FROM Professionals AS T1 JOIN Treatments AS T2 WHERE cost_of_treatment  <  ( SELECT avg(cost_of_treatment) FROM Treatments )':
    'SELECT DISTINCT T1.first_name ,  T1.last_name FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id WHERE cost_of_treatment  <  ( SELECT avg(cost_of_treatment) FROM Treatments )',
    'SELECT T2.apt_number ,  T1.booking_start_date ,  T1.booking_start_date FROM Apartment_Bookings AS T1 JOIN Apartments AS T2 ON T1.apt_id  =  T2.apt_id':
    'SELECT T2.apt_number ,  T1.booking_start_date ,  T1.booking_end_date FROM Apartment_Bookings AS T1 JOIN Apartments AS T2 ON T1.apt_id  =  T2.apt_id',
    'SELECT T1.booking_start_date ,  T1.booking_start_date FROM Apartment_Bookings AS T1 JOIN Apartments AS T2 ON T1.apt_id  =  T2.apt_id WHERE T2.apt_type_code  =  "Duplex"':
    'SELECT T1.booking_start_date ,  T1.booking_end_date FROM Apartment_Bookings AS T1 JOIN Apartments AS T2 ON T1.apt_id  =  T2.apt_id WHERE T2.apt_type_code  =  "Duplex"',
    'SELECT T1.booking_start_date ,  T1.booking_start_date FROM Apartment_Bookings AS T1 JOIN Apartments AS T2 ON T1.apt_id  =  T2.apt_id WHERE T2.bedroom_count  >  2':
    'SELECT T1.booking_start_date ,  T1.booking_end_date FROM Apartment_Bookings AS T1 JOIN Apartments AS T2 ON T1.apt_id  =  T2.apt_id WHERE T2.bedroom_count  >  2',
    'SELECT T2.guest_first_name ,  T1.booking_start_date ,  T1.booking_start_date FROM Apartment_Bookings AS T1 JOIN Guests AS T2 ON T1.guest_id  =  T2.guest_id':
    'SELECT T2.guest_first_name ,  T1.booking_start_date ,  T1.booking_end_date FROM Apartment_Bookings AS T1 JOIN Guests AS T2 ON T1.guest_id  =  T2.guest_id',
    'SELECT T1.booking_start_date ,  T1.booking_start_date FROM Apartment_Bookings AS T1 JOIN Guests AS T2 ON T1.guest_id  =  T2.guest_id WHERE T2.gender_code  =  "Female"':
    'SELECT T1.booking_start_date ,  T1.booking_end_date FROM Apartment_Bookings AS T1 JOIN Guests AS T2 ON T1.guest_id  =  T2.guest_id WHERE T2.gender_code  =  "Female"',
    'SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 WHERE T2.year  =  1960 INTERSECT SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 WHERE T2.year  =  1961':
    'SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 ON T1.player_id  =  T2.player_id WHERE T2.year  =  1960 INTERSECT SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 ON T1.player_id  =  T2.player_id WHERE T2.year  =  1961',
    'SELECT DISTINCT t1.product_name FROM products AS t1 JOIN complaints AS t2 ON t1.product_id  =  t2.product_id JOIN customers AS t3 GROUP BY t3.customer_id ORDER BY count(*) LIMIT 1':
    'SELECT DISTINCT t1.product_name FROM products AS t1 JOIN complaints AS t2 ON t1.product_id  =  t2.product_id JOIN customers AS t3 ON t2.customer_id  = t3.customer_id GROUP BY t3.customer_id ORDER BY count(*) LIMIT 1',
    "SELECT DISTINCT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T2.actid WHERE T3.activity_name  =  'Canoeing' OR T3.activity_name  =  'Kayaking'":
    "SELECT DISTINCT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T3.actid WHERE T3.activity_name  =  'Canoeing' OR T3.activity_name  =  'Kayaking'",
    "SELECT lname FROM faculty WHERE rank  =  'Professor' EXCEPT SELECT DISTINCT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T2.actid WHERE T3.activity_name  =  'Canoeing' OR T3.activity_name  =  'Kayaking'":
    "SELECT lname FROM faculty WHERE rank  =  'Professor' EXCEPT SELECT DISTINCT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T3.actid WHERE T3.activity_name  =  'Canoeing' OR T3.activity_name  =  'Kayaking'",
    "SELECT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T2.actid WHERE T3.activity_name  =  'Canoeing' INTERSECT SELECT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T2.actid WHERE T3.activity_name  =  'Kayaking'":
    "SELECT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T3.actid WHERE T3.activity_name  =  'Canoeing' INTERSECT SELECT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T3.actid WHERE T3.activity_name  =  'Kayaking'",
    "SELECT T1.stuid FROM participates_in AS T1 JOIN activity AS T2 ON T2.actid  =  T2.actid WHERE T2.activity_name  =  'Canoeing' INTERSECT SELECT T1.stuid FROM participates_in AS T1 JOIN activity AS T2 ON T2.actid  =  T2.actid WHERE T2.activity_name  =  'Kayaking'":
    "SELECT T1.stuid FROM participates_in AS T1 JOIN activity AS T2 ON T1.actid  =  T2.actid WHERE T2.activity_name  =  'Canoeing' INTERSECT SELECT T1.stuid FROM participates_in AS T1 JOIN activity AS T2 ON T1.actid  =  T2.actid WHERE T2.activity_name  =  'Kayaking'",
}
