import React from 'react';
import ReactDOM from 'react-dom/client';
import Search from '../src/routes/Search';
import SearchResults from '../src/routes/SearchResults';
import SearchResult from './routes/SearchResult';
import ErrorPage from './ErrorPage';

import './index.css'
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";




const router = createBrowserRouter([
  {
    path: "/",
    element: <Search/>,
    errorElement:<ErrorPage/>
  },{
    path: "/searchResults",
    element: <SearchResults/>,
    errorElement:<ErrorPage/>,
    children: [
      {
        path: ":searchId",
        element: <SearchResult />,
        errorElement:<ErrorPage/>
      },
    ],
  }
]);

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
