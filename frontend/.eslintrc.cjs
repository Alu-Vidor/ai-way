module.exports = {
  root: true,
  env: {
    browser: true,
    es2020: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended'
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    project: './tsconfig.json',
    tsconfigRootDir: __dirname,
    ecmaFeatures: {
      jsx: true
    }
  },
  plugins: ['@typescript-eslint'],
  rules: {
    'react-refresh/only-export-components': 'off'
  },
  settings: {
    react: {
      version: 'detect'
    }
  },
  ignorePatterns: ['dist', 'node_modules', 'tailwind.config.ts', 'vite.config.ts']
};
