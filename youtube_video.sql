CREATE DATABASE IF NOT EXISTS youtube_videos;
USE youtube_videos;
DROP TABLE videos ;
/*
LOAD DATA INFILE '/tmp/mydata.txt' INTO TABLE videos;

*/
 CREATE TABLE IF NOT EXISTS videos(
	vid INT NOT NULL,
	age INT, 
	uploaderID INT NOT NULL,
	category VARCHAR(255),
	length decimal,
	rate INT, 
	ratings INT, 
	comments VARCHAR(255), 
	views DECIMAL,
	PRIMARY KEY (vid,uploaderID)
);

 CREATE TABLE IF NOT EXISTS uploader(
	uploaderID INT NOT NULL,
	PRIMARY KEY (uploaderID)
);

 CREATE TABLE IF NOT EXISTS uploaded_by(
    vid INT NOT NULL,
    uploaderID INT NOT NULL,
    PRIMARY KEY (vid, uploaderID),
    FOREIGN KEY (vid) REFERENCES  videos(vid) on delete cascade,
    FOREIGN KEY (uploaderID)  REFERENCES  uploader (uploaderID) on delete cascade
);

 CREATE TABLE IF NOT EXISTS related_to(
    vid INT NOT NULL AUTO_INCREMENT,
    PRIMARY KEY(vid),
    FOREIGN KEY(vid) REFERENCES videos(vid)
);