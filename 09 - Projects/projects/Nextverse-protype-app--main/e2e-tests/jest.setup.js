// Jest setup file
require('dotenv').config();
jest.setTimeout(parseInt(process.env.TIMEOUT || '30000', 10));
