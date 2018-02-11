
" colors
colorscheme badwolf         " awesome colorscheme
"colorscheme github

syntax enable           " enable syntax processing

"spaces and tabs
set tabstop=4       " number of visual spaces per TAB

set softtabstop=4   " number of spaces in tab when editing

set expandtab       " tabs are spaces, turn 1 tab into 4 spaces

" UI config
"set number              " show line numbers
set ruler               " Show the line and column number of the cursor
" position, eparated by a comma.

set showcmd             " show command in bottom bar

set cursorline          " highlight current line

"set autoindent      " Copy indent from current line when starting a new line

filetype indent plugin on      " load filetype-specific indent files

set wildmenu            " visual autocomplete for command menu

set lazyredraw          " redraw only when we need to.

set showmatch           " highlight matching [{()}]

" searching
set incsearch           " search as characters are entered
set hlsearch            " highlight matches
" turn off search highlight
nnoremap <leader><space> :nohlsearch<CR>

" Folding
set foldenable          " enable folding
set foldlevelstart=10   " open most folds by default
set foldnestmax=10      " 10 nested fold max
" space open/closes folds
nnoremap <space> za
set foldmethod=indent   " fold based on indent level

" Movement

" move vertically by visual line
nnoremap j gj
nnoremap k gk

" move to beginning/end of line
nnoremap B ^
nnoremap E $

" $/^ doesn't do anything
"nnoremap $ <nop>
"nnoremap ^ <nop>

" highlight last inserted text
nnoremap gV `[v`]

" Mouse
" Enable mouse in all modes
set mouse=a
set bs=2

" leader shortcuts
let mapleader=","       " leader is comma

" " Autogroups
" "augroup configgroup
" "        autocmd!
" "        autocmd VimEnter * highlight clear SignColumn
" "        autocmd BufWritePre *.php,*.py,*.js,*.txt,*.hs,*.java,*.md
" "                    \:call <SID>StripTrailingWhitespaces()
" "        autocmd FileType java setlocal noexpandtab
" "        autocmd FileType java setlocal list
" "        autocmd FileType java setlocal listchars=tab:+\ ,eol:-
" "        autocmd FileType java setlocal formatprg=par\ -w80\ -T4
" "        autocmd FileType php setlocal expandtab
" "        autocmd FileType php setlocal list
" "        autocmd FileType php setlocal listchars=tab:+\ ,eol:-
" "        autocmd FileType php setlocal formatprg=par\ -w80\ -T4
" "        autocmd FileType ruby setlocal tabstop=2
" "        autocmd FileType ruby setlocal shiftwidth=2
" "        autocmd FileType ruby setlocal softtabstop=2
" "        autocmd FileType ruby setlocal commentstring=#\ %s
" "        autocmd FileType python setlocal commentstring=#\ %s
" "        autocmd BufEnter *.cls setlocal filetype=java
" "        autocmd BufEnter *.zsh-theme setlocal filetype=zsh
" "        autocmd BufEnter Makefile setlocal noexpandtab
" "        autocmd BufEnter *.sh setlocal tabstop=2
" "        autocmd BufEnter *.sh setlocal shiftwidth=2
" "        autocmd BufEnter *.sh setlocal softtabstop=2
" "augroup END
"

