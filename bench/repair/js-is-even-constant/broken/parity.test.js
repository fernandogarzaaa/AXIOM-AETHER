const assert = require('assert');
const isEven = require('./parity');
assert.strictEqual(isEven(4), true);
assert.strictEqual(isEven(7), false);
assert.strictEqual(isEven(0), true);
