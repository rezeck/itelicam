/* eslint-env node, mocha */

// import node_modules
import chai, { expect } from "chai";

// import custom modules
import app from "./index";

// Test generation of stories
describe("START EXPRESS SERVER", function() {
  it("should start the express server", done => {
    expect(app).to.be.a("function");
    done();
  });
});
