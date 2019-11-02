/* eslint-disable import/no-extraneous-dependencies */

// import node_modules
import bodyParser from "body-parser";
import cors from "cors";
import express from "express";
import http from "http";
import morgan from "morgan";

// Define Babel 6 regeneratorRuntime
// https://stackoverflow.com/questions/33527653/babel-6-regeneratorruntime-is-not-defined
import "@babel/register";
import "core-js";

// On Windows, !process.env.PWD is not used, use cwd instead
if (!process.env.PWD) {
  process.env.PWD = process.cwd();
}

// Set dotenv config and load variables from .env files
require("dotenv").config({
  path: `${process.env.PWD}/.env.${process.env.NODE_ENV}`
});

// init express server
const app = express();
app.server = http.createServer(app);

// logger
app.use(morgan("dev"));

// 3rd party middleware
app.use(
  cors({
    exposedHeaders: ["Link"]
  })
);

// parse request body
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

app.server.listen(process.env.PORT || 8080, () => {
  // eslint-disable-next-line no-console
  console.log(`Started on port ${app.server.address().port}`);
});

export default app;
