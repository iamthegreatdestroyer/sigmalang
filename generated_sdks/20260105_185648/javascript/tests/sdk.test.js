/**
 * Tests for SigmaLang JavaScript SDK
 */

const { SigmaLang } = require('../dist/index.js');

describe('SigmaLang', () => {
  let client;

  beforeEach(() => {
    client = new SigmaLang('test-key');
  });

  test('should create client', () => {
    expect(client).toBeDefined();
  });

  // Add more tests when API is available
});
